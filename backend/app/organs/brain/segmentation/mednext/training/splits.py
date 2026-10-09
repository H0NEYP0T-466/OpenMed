"""Split construction and split-level hygiene.

Policy
------
**OpenMed** — filenames encode lesion and slice
(``T1C+ - Cystic glioblastoma occipital , ventricle 005``). Stripping the
sequence prefix and trailing slice number gives a lesion/series key, so every
slice of a lesion lands in one split. The key is a *proxy* for a patient id: it
can merge distinct patients (safe) but cannot be proven to never split one.

**BTSC** — filenames are ``1.png ... 3064.png`` with no patient id. Adjacent
indices are strongly correlated (measured median 0.93 vs 0.54 for random
pairs), so the ordering is cut into contiguous blocks, and each cut is snapped
to the weakest adjacent-slice similarity inside a small window — the most
likely scan boundary. Blocks linked by near-duplicate slices are merged. After
splitting, any validation/test slice that still sits next to a highly similar
training slice is purged from evaluation.

Groups are partitioned, never slices. Many random partitions are scored on how
well they hit the target sizes and how closely the lesion-size distribution of
val/test matches train; the best one wins (seeded, deterministic).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.stats import ks_2samp

from .audit import (
    CheckLog,
    SourceAudit,
    UnionFind,
    _popcount,
    hamming_pairs,
    paths_for,
    pixel_mae,
)

logger = logging.getLogger("splits")

_SEQ_PREFIX = re.compile(r"^(T1C\+|T1|T2)\s*-\s*")
_TRAILING_NUMBER = re.compile(r"\s+\d+$")
SPLITS = ("train", "val", "test")


def openmed_group_key(stem: str) -> str:
    """Lesion/series key for an OpenMed filename. Not a verified patient key."""
    return _SEQ_PREFIX.sub("", _TRAILING_NUMBER.sub("", stem))


def openmed_group_lookup(root: str):
    """``stem -> group`` callable; prefers ``manifest.csv`` when the dataset ships one."""
    manifest = os.path.join(root, "manifest.csv")
    mapping: dict[str, str] = {}
    if os.path.isfile(manifest):
        try:
            with open(manifest, newline="") as handle:
                for row in csv.DictReader(handle):
                    stem = (row.get("stem") or "").strip()
                    group = (row.get("group") or "").strip()
                    if stem and group:
                        mapping[stem] = group
        except (OSError, csv.Error) as exc:
            logger.warning("manifest.csv unreadable (%s) — using filename keys", exc)
    if mapping:
        logger.info("[openmed] group keys from manifest.csv (%d entries)", len(mapping))

        def from_manifest(stem: str) -> str:
            return mapping.get(stem) or openmed_group_key(stem)

        return from_manifest
    logger.info("[openmed] no manifest.csv — group keys derived from filenames")
    return openmed_group_key


# ─────────────────────────────────────────────────────────────────────────────
# Records and manifest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Record:
    source: str
    stem: str
    image_path: str
    mask_path: str
    group: str
    fg_frac: float
    gray_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "stem": self.stem, "image": self.image_path,
                "mask": self.mask_path, "group": self.group,
                "fg_frac": self.fg_frac, "gray_mean": self.gray_mean}

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Record:
        return cls(row["source"], row["stem"], row["image"], row["mask"], row["group"],
                   row.get("fg_frac", 0.0), row.get("gray_mean", 0.0))


@dataclass
class SplitManifest:
    train: list[Record] = field(default_factory=list)
    val: list[Record] = field(default_factory=list)
    test: list[Record] = field(default_factory=list)
    purged: list[dict[str, str]] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)

    def part(self, name: str) -> list[Record]:
        return getattr(self, name)

    def all_records(self) -> list[Record]:
        return [*self.train, *self.val, *self.test]

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        for name in SPLITS:
            with open(os.path.join(out_dir, f"{name}.json"), "w") as handle:
                json.dump([r.to_dict() for r in self.part(name)], handle, indent=1)
        with open(os.path.join(out_dir, "split_manifest.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["split", "source", "stem", "group", "fg_frac"])
            for name in SPLITS:
                for rec in self.part(name):
                    writer.writerow([name, rec.source, rec.stem, rec.group, f"{rec.fg_frac:.6f}"])
        with open(os.path.join(out_dir, "purged_eval_slices.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["split", "source", "stem", "reason"])
            for row in self.purged:
                writer.writerow([row["split"], row["source"], row["stem"], row["reason"]])

    @classmethod
    def load(cls, out_dir: str) -> SplitManifest:
        manifest = cls()
        for name in SPLITS:
            with open(os.path.join(out_dir, f"{name}.json")) as handle:
                setattr(manifest, name, [Record.from_dict(r) for r in json.load(handle)])
        return manifest


def build_records(audit: SourceAudit) -> list[Record]:
    image_paths, mask_paths = paths_for(audit.spec, audit.kept)
    group_of = audit.spec.group_of
    records = []
    for stem, ip, mp in zip(audit.kept, image_paths, mask_paths, strict=True):
        row = audit.rows[stem]
        records.append(Record(
            source=audit.spec.name, stem=stem, image_path=ip, mask_path=mp,
            group=str(group_of(stem)) if group_of else stem,
            fg_frac=float(row["fg_frac"]), gray_mean=float(row["gray_mean"]),
        ))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# BTSC: snapped contiguous blocks
# ─────────────────────────────────────────────────────────────────────────────

def _numeric_order(stems: list[str]) -> list[str]:
    def key(stem: str):
        try:
            return (0, int(stem), stem)
        except ValueError:
            return (1, 0, stem)
    return sorted(stems, key=key)


def _standardised(thumbs: dict[str, np.ndarray], order: list[str]) -> np.ndarray:
    feats = np.stack([thumbs[s].astype(np.float32).ravel() for s in order])
    feats -= feats.mean(axis=1, keepdims=True)
    feats /= feats.std(axis=1, keepdims=True) + 1e-6
    return feats


def adjacent_similarity(thumbs: dict[str, np.ndarray], order: list[str]) -> np.ndarray:
    """Correlation between each slice and the next one in ``order``."""
    feats = _standardised(thumbs, order)
    return (feats[:-1] * feats[1:]).mean(axis=1)


def snapped_blocks(
    order: list[str], adjacent: np.ndarray, block_size: int, window: int
) -> dict[str, str]:
    """Cut ``order`` into runs of about ``block_size``, each cut at a similarity dip."""
    n = len(order)
    cuts: list[int] = []
    start = 0
    while start + block_size + window < n:
        target = start + block_size
        low = max(start + block_size // 2, target - window)
        high = min(n - 2, target + window)
        cut = low + int(np.argmin(adjacent[low:high + 1]))
        cuts.append(cut + 1)
        start = cut + 1
    boundaries = [0, *cuts, n]
    if len(boundaries) > 2 and boundaries[-1] - boundaries[-2] < block_size // 2:
        boundaries.pop(-2)

    assignment: dict[str, str] = {}
    for block_id, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        for stem in order[a:b]:
            assignment[stem] = f"block_{block_id:05d}"
    return assignment


def btsc_adjacency_evidence(adjacent: np.ndarray, thumbs: dict[str, np.ndarray],
                            order: list[str], seed: int) -> dict[str, Any]:
    """Quantify whether index adjacency really tracks scan membership."""
    feats = _standardised(thumbs, order)
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(order), 5000)
    b = rng.integers(0, len(order), 5000)
    random_corr = (feats[a] * feats[b]).mean(axis=1)
    return {
        "adjacent_corr_median": float(np.median(adjacent)),
        "adjacent_corr_p10": float(np.percentile(adjacent, 10)),
        "random_corr_median": float(np.median(random_corr)),
        "random_corr_p90": float(np.percentile(random_corr, 90)),
        "frac_adjacent_above_0.8": float((adjacent > 0.8).mean()),
        "frac_random_above_0.8": float((random_corr > 0.8).mean()),
    }


def merge_groups_by_twins(
    records: list[Record],
    audit: SourceAudit,
    max_hamming: int,
    cap_frac: float,
) -> tuple[list[Record], dict[str, Any]]:
    """Union groups connected by near-duplicate slices, refusing giant merges."""
    stems = [r.stem for r in records]
    hashes = np.array([audit.rows[s]["phash"] for s in stems], dtype=np.uint64)
    pairs = hamming_pairs(hashes, max_hamming)

    group_ids = sorted({r.group for r in records})
    index = {g: i for i, g in enumerate(group_ids)}
    sizes = defaultdict(int)
    for rec in records:
        sizes[rec.group] += 1
    union = UnionFind(len(group_ids))
    component_size = {i: sizes[g] for g, i in index.items()}
    cap = int(cap_frac * len(records))

    merged = refused = 0
    for i, j, _ in sorted(pairs, key=lambda p: p[2]):
        ga, gb = index[records[i].group], index[records[j].group]
        ra, rb = union.find(ga), union.find(gb)
        if ra == rb:
            continue
        if component_size[ra] + component_size[rb] > cap:
            refused += 1
            continue
        union.union(ra, rb)
        root = union.find(ra)
        component_size[root] = component_size[ra] + component_size[rb]
        merged += 1

    for rec in records:
        rec.group = f"{audit.spec.name}:g{union.find(index[rec.group]):05d}"
    return records, {"twin_pairs": len(pairs), "merges": merged, "refused_over_cap": refused,
                     "cap_slices": cap}


# ─────────────────────────────────────────────────────────────────────────────
# Partitioning
# ─────────────────────────────────────────────────────────────────────────────

def _ks(a: list[float], b: list[float]) -> float:
    if len(a) < 5 or len(b) < 5:
        return 1.0
    return float(ks_2samp(a, b).statistic)


def _greedy_partition(
    keys: list[str], groups: dict[str, list[Record]], val_target: int, test_target: int
) -> tuple[list[Record], list[Record], list[Record]]:
    val: list[Record] = []
    test: list[Record] = []
    train: list[Record] = []
    for key in keys:
        bucket = groups[key]
        if len(val) < val_target:
            val.extend(bucket)
        elif len(test) < test_target:
            test.extend(bucket)
        else:
            train.extend(bucket)
    return train, val, test


def partition_groups(
    records: list[Record], val_frac: float, test_frac: float, seed: int, tries: int,
) -> tuple[list[Record], list[Record], list[Record], dict[str, Any]]:
    """Best of ``tries`` random group partitions by size error and distribution match."""
    groups: dict[str, list[Record]] = defaultdict(list)
    for rec in records:
        groups[rec.group].append(rec)
    base_keys = sorted(groups)
    total = len(records)
    val_target, test_target = int(total * val_frac), int(total * test_frac)
    rng = random.Random(seed)

    best: Optional[tuple[float, list[Record], list[Record], list[Record]]] = None
    for _ in range(max(1, tries)):
        keys = list(base_keys)
        rng.shuffle(keys)
        train, val, test = _greedy_partition(keys, groups, val_target, test_target)
        if not train or not val or not test:
            continue
        score = 0.0
        if val_frac:
            score += abs(len(val) / total - val_frac) / val_frac
        if test_frac:
            score += abs(len(test) / total - test_frac) / test_frac
        tr_fg = [r.fg_frac for r in train]
        tr_gm = [r.gray_mean for r in train]
        for part in (val, test):
            score += _ks(tr_fg, [r.fg_frac for r in part])
            score += 0.5 * _ks(tr_gm, [r.gray_mean for r in part])
        if best is None or score < best[0]:
            best = (score, train, val, test)

    if best is None:
        raise RuntimeError(
            f"Could not build a non-empty train/val/test partition from {len(base_keys)} groups."
        )
    score, train, val, test = best
    return train, val, test, {"score": round(score, 4), "tries": tries, "n_groups": len(base_keys)}


# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────

def build_splits(
    audits: dict[str, SourceAudit], cfg: Any
) -> SplitManifest:
    manifest = SplitManifest()
    details: dict[str, Any] = {}

    for name, audit in audits.items():
        records = build_records(audit)
        detail: dict[str, Any] = {"n_records": len(records)}

        if audit.spec.group_of is None:
            order = _numeric_order([r.stem for r in records])
            adjacent = adjacent_similarity(audit.thumbs, order)
            detail["adjacency_evidence"] = btsc_adjacency_evidence(
                adjacent, audit.thumbs, order, cfg.seed
            )
            assignment = snapped_blocks(order, adjacent, cfg.btsc_block_size, cfg.btsc_snap_window)
            for rec in records:
                rec.group = f"{name}:{assignment[rec.stem]}"
            detail["n_blocks_before_merge"] = len(set(assignment.values()))
            evidence = detail["adjacency_evidence"]
            logger.info(
                "[%s] adjacency evidence: adjacent-slice corr median %.3f vs random %.3f; "
                "%d snapped blocks",
                name, evidence["adjacent_corr_median"], evidence["random_corr_median"],
                detail["n_blocks_before_merge"],
            )

        records, merge = merge_groups_by_twins(
            records, audit, cfg.phash_hamming_max, cfg.group_merge_cap_frac
        )
        detail["twin_merge"] = merge
        detail["n_groups"] = len({r.group for r in records})
        sizes = np.array(list(_group_sizes(records).values()))
        detail["group_sizes"] = {"median": float(np.median(sizes)), "max": int(sizes.max()),
                                 "min": int(sizes.min())}

        mode = cfg.btsc_mode if audit.spec.group_of is None else "grouped"
        if mode == "train_only":
            manifest.train.extend(records)
            detail["partition"] = {"mode": "train_only"}
        elif mode == "external_test":
            manifest.test.extend(records)
            detail["partition"] = {"mode": "external_test"}
        else:
            train, val, test, info = partition_groups(
                records, cfg.val_frac, cfg.test_frac, cfg.seed, cfg.split_search_tries
            )
            manifest.train.extend(train)
            manifest.val.extend(val)
            manifest.test.extend(test)
            detail["partition"] = {"mode": mode, **info}
        details[name] = detail
        logger.info("[%s] %s", name, json.dumps({k: v for k, v in detail.items()
                                                 if k != "adjacency_evidence"}))

    manifest.report = {"seed": cfg.seed, "sources": details}
    return manifest


def _group_sizes(records: list[Record]) -> dict[str, int]:
    sizes: dict[str, int] = defaultdict(int)
    for rec in records:
        sizes[rec.group] += 1
    return dict(sizes)


# ─────────────────────────────────────────────────────────────────────────────
# Split verification (and the purges it can justify)
# ─────────────────────────────────────────────────────────────────────────────

def _counts(manifest: SplitManifest) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for name in SPLITS:
        counts: dict[str, int] = defaultdict(int)
        for rec in manifest.part(name):
            counts[rec.source] += 1
        out[name] = dict(counts)
    return out


def verify_split(
    manifest: SplitManifest, audits: dict[str, SourceAudit], cfg: Any, log: CheckLog,
) -> dict[str, Any]:
    """Run V1-V8. Purges evaluation slices that provably duplicate training data."""
    purged: list[dict[str, str]] = []

    def purge(split: str, rec: Record, reason: str) -> None:
        if rec in manifest.part(split):
            manifest.part(split).remove(rec)
            purged.append({"split": split, "source": rec.source, "stem": rec.stem, "reason": reason})

    # V1 group leakage
    seen: dict[tuple[str, str], str] = {}
    leaks = []
    for name in SPLITS:
        for rec in manifest.part(name):
            key = (rec.source, rec.group)
            if key not in seen:
                seen[key] = name
            elif seen[key] != name:
                leaks.append({"source": rec.source, "group": rec.group,
                              "first": seen[key], "second": name})
    log.add("V1", "group leakage", "PASS" if not leaks else "FAIL",
            f"{len(leaks)} groups appear in more than one split", examples=leaks[:10])

    # V2 stem overlap
    owner: dict[tuple[str, str], str] = {}
    stem_clash = 0
    for name in SPLITS:
        for rec in manifest.part(name):
            key = (rec.source, rec.stem)
            if key in owner and owner[key] != name:
                stem_clash += 1
            owner[key] = name
    log.add("V2", "stem overlap", "PASS" if not stem_clash else "FAIL",
            f"{stem_clash} stems appear in more than one split")

    # V3 exact-content leakage
    digests: dict[str, set[str]] = {n: set() for n in SPLITS}
    for name in SPLITS:
        for rec in manifest.part(name):
            digests[name].add(audits[rec.source].rows[rec.stem]["sha_pixels"])
    exact = {
        "val": len(digests["val"] & digests["train"]),
        "test": len(digests["test"] & digests["train"]),
        "val_test": len(digests["val"] & digests["test"]),
    }
    log.add("V3", "exact content leakage", "PASS" if not any(exact.values()) else "FAIL",
            f"decoded-pixel overlap train/val {exact['val']}, train/test {exact['test']}, "
            f"val/test {exact['val_test']}", **exact)

    # V4 near-duplicate twins across splits (pHash, pixel-confirmed)
    twins_purged = 0
    info_pairs = 0
    train_by_source: dict[str, list[Record]] = defaultdict(list)
    for rec in manifest.train:
        train_by_source[rec.source].append(rec)
    for split in ("val", "test"):
        for source, audit in audits.items():
            ev = [r for r in manifest.part(split) if r.source == source]
            tr = train_by_source.get(source, [])
            if not ev or not tr:
                continue
            ev_h = np.array([audit.rows[r.stem]["phash"] for r in ev], dtype=np.uint64)
            tr_h = np.array([audit.rows[r.stem]["phash"] for r in tr], dtype=np.uint64)

            for start in range(0, len(ev_h), 512):
                dist = _popcount(ev_h[start:start + 512, None] ^ tr_h[None, :])
                rows_, cols_ = np.nonzero(dist <= cfg.phash_hamming_max)
                info_pairs += len(rows_)
                for r_, c_ in zip(rows_, cols_, strict=True):
                    if dist[r_, c_] > cfg.phash_dup_hamming:
                        continue
                    a, b = ev[start + int(r_)], tr[int(c_)]
                    mae = pixel_mae(a.image_path, b.image_path)
                    if mae is not None and mae < cfg.dup_pixel_mae:
                        before = len(purged)
                        purge(split, a, f"near-identical to training slice {b.stem} (MAE {mae:.2f})")
                        twins_purged += int(len(purged) > before)
    log.add("V4", "near-dup twins", "PASS" if not twins_purged else "WARN",
            f"{twins_purged} val/test slices were pixel-near-identical to a training slice and "
            f"were purged from evaluation; {info_pairs} further pHash<= {cfg.phash_hamming_max} "
            "pairs are merely similar-looking neighbouring anatomy (kept)",
            purged=twins_purged, similar_pairs=info_pairs)

    # V5 BTSC adjacency exposure and purge
    adjacency: dict[str, Any] = {"applicable": False}
    for source, audit in audits.items():
        if audit.spec.group_of is not None:
            continue
        order = _numeric_order([r.stem for r in manifest.all_records() if r.source == source])
        if len(order) < 3:
            continue
        split_of = {r.stem: n for n in SPLITS for r in manifest.part(n) if r.source == source}
        rec_of = {r.stem: r for n in SPLITS for r in manifest.part(n) if r.source == source}
        feats = _standardised(audit.thumbs, order)
        crossing = 0
        crossing_corr: list[float] = []
        exposed: set[str] = set()
        position = {s: i for i, s in enumerate(order)}
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            if split_of[a] != split_of[b]:
                crossing += 1
                crossing_corr.append(float((feats[i] * feats[i + 1]).mean()))
        for stem, split in split_of.items():
            if split == "train":
                continue
            i = position[stem]
            for off in range(-cfg.btsc_purge_radius, cfg.btsc_purge_radius + 1):
                j = i + off
                if off == 0 or j < 0 or j >= len(order):
                    continue
                if split_of[order[j]] == "train":
                    corr = float((feats[i] * feats[j]).mean())
                    if corr >= cfg.btsc_purge_corr:
                        exposed.add(stem)
                        break
        for stem in sorted(exposed):
            purge(split_of[stem], rec_of[stem],
                  f"index-adjacent to a highly similar training slice (corr >= {cfg.btsc_purge_corr})")
        adjacency = {
            "applicable": True, "source": source,
            "adjacent_pairs_crossing_a_split": crossing,
            "crossing_corr_median": float(np.median(crossing_corr)) if crossing_corr else None,
            "eval_slices_purged": len(exposed),
            "interpretation": (
                "Residual risk of the block split. Crossing pairs with high similarity are "
                "likely the same scan; those evaluation slices were purged. Without a patient "
                "id this cannot be driven to a proof of zero."),
        }
        median = adjacency["crossing_corr_median"]
        log.add("V5", f"{source} adjacency", "WARN" if crossing else "PASS",
                f"{crossing} adjacent index pairs cross a split boundary "
                f"(median corr {median if median is None else round(median, 3)}); "
                f"{len(exposed)} val/test slices purged for sitting beside a near-identical "
                "training slice", **{k: v for k, v in adjacency.items() if k != "interpretation"})

    # V6 distribution parity
    parity: dict[str, Any] = {}
    for source in audits:
        tr = [r for r in manifest.train if r.source == source]
        for split in ("val", "test"):
            ev = [r for r in manifest.part(split) if r.source == source]
            if len(ev) >= 5 and len(tr) >= 5:
                parity[f"{source}:{split}"] = {
                    "ks_fg_frac": round(_ks([r.fg_frac for r in tr], [r.fg_frac for r in ev]), 4),
                    "ks_gray_mean": round(_ks([r.gray_mean for r in tr], [r.gray_mean for r in ev]), 4),
                    "n": len(ev),
                }
    worst = max((v["ks_fg_frac"] for v in parity.values()), default=0.0)
    log.add("V6", "distribution parity", "PASS" if worst < 0.15 else "WARN",
            f"KS(lesion-size) train-vs-eval worst {worst:.3f} (0 identical, >0.15 notable)",
            parity=parity)

    # V7 proportions and presence
    counts = _counts(manifest)
    total = {n: sum(c.values()) for n, c in counts.items()}
    grand = max(1, sum(total.values()))
    log.add("V7", "split sizes", "INFO",
            "; ".join(f"{n}={total[n]} ({total[n] / grand:.1%}) {counts[n]}" for n in SPLITS),
            counts=counts, total=total)

    # V8 minimum evaluation size
    too_small = [n for n in ("val", "test") if total[n] and total[n] < 100]
    empty_val = total["val"] == 0
    log.add("V8", "evaluation size", "FAIL" if empty_val else ("WARN" if too_small else "PASS"),
            f"val={total['val']} test={total['test']} slices",
            too_small=too_small)

    manifest.purged.extend(purged)
    manifest.report.update({
        "counts": counts, "total": total, "group_leaks": leaks, "n_group_leaks": len(leaks),
        "exact_leakage": exact, "twins_purged": twins_purged,
        "adjacency": adjacency, "parity": parity,
        "caveat": (
            "OpenMed groups are filename-derived lesion/series keys; BTSC groups are snapped "
            "contiguous blocks. Neither is a verified patient identifier. No group appears in "
            "more than one split; residual BTSC risk is quantified under 'adjacency'."),
    })
    return manifest.report
