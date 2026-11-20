"""Stage B: leak-free grouped split.

Clusters, never images, are partitioned. The unit is a duplicate cluster (union-find
over SHA-256 of bytes and pixels plus pixel-confirmed pHash), so an image and its
re-encoded or near-identical copy always land in the same split. A cluster that
spans two splits is a hard assertion failure.

Target proportions are the shipped 64 / 16 / 20. Strata are ``(class, tree
membership)``: class so each split holds COVID-19 / Non-COVID / Normal in
proportion, tree membership because it decides head-B supervision and every split
needs supervised positives and negatives to be evaluated at all.

The report is a deliverable: counts before and after, cluster size distribution,
how many images moved against the shipped split, and the cross-split duplicate
pairs before and after, each *recomputed* on the final partition rather than
assumed to be zero.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..labels import CLASSES
from .catalog import Record
from .clusters import ClusterResult
from .config import SPLITS, TrainConfig

logger = logging.getLogger("splits")

MANIFEST_COLUMNS = (
    "uid", "filename", "image_path", "lung_mask_path", "infection_mask_path", "class", "tree",
    "in_infection_tree", "opacity_supervised", "cluster_id", "cluster_size", "split",
    "shipped_split",
)


@dataclass
class SplitResult:
    split_of: dict[int, str]
    records: dict[str, list[Record]]
    cluster_of: dict[int, int]
    report: dict[str, Any]

    def part(self, name: str) -> list[Record]:
        return self.records[name]


def stratum(record: Record) -> tuple[str, bool]:
    return record.cls, record.in_infection_tree


def assert_cluster_purity(cluster_of: dict[int, int], split_of: dict[int, str]) -> None:
    """Raise if any cluster has members in more than one split."""
    seen: dict[int, set[str]] = defaultdict(set)
    for uid, split in split_of.items():
        seen[cluster_of[uid]].add(split)
    impure = {cid: sorted(s) for cid, s in seen.items() if len(s) > 1}
    if impure:
        sample = dict(list(impure.items())[:5])
        raise AssertionError(
            f"{len(impure)} duplicate cluster(s) span more than one split, e.g. {sample}. "
            "A split that shares a source image between train and test is a leak."
        )


def cross_split_counts(
    clusters: ClusterResult, uids: set[int], split_of: dict[int, str]
) -> dict[str, Any]:
    """Duplicate pairs and clusters that straddle a split boundary, on ``uids``."""
    def crossing(pairs: list[tuple[int, int]]) -> int:
        return sum(1 for a, b in pairs
                   if a in uids and b in uids and split_of[a] != split_of[b])

    near_cross = [(k, v) for k, v in clusters.near_pairs.items()
                  if k[0] in uids and k[1] in uids and split_of[k[0]] != split_of[k[1]]]
    spans: dict[int, set[str]] = defaultdict(set)
    subject_spans: dict[str, set[str]] = defaultdict(set)
    for uid in uids:
        spans[clusters.cluster_of[uid]].add(split_of[uid])
        if uid in clusters.subject_of:
            subject_spans[clusters.subject_of[uid]].add(split_of[uid])
    return {
        "subject_groups_spanning_splits": sum(1 for s in subject_spans.values() if len(s) > 1),
        "exact_pixel_pairs": crossing(clusters.exact_pixel_pairs),
        "exact_byte_pairs": crossing(clusters.exact_byte_pairs),
        "phash_confirmed_pairs": len(near_cross),
        "phash_identical_hash_pairs": sum(1 for _, v in near_cross if int(v["hamming"]) == 0),
        "clusters_spanning_splits": sum(1 for s in spans.values() if len(s) > 1),
    }


def _greedy_assign(
    survivors: list[Record], cluster_of: dict[int, int], cfg: TrainConfig
) -> dict[int, str]:
    strata = sorted({stratum(r) for r in survivors}, key=lambda s: (CLASSES.index(s[0]), s[1]))
    sidx = {s: i for i, s in enumerate(strata)}
    totals = np.zeros(len(strata))
    members: dict[int, list[Record]] = defaultdict(list)
    for r in survivors:
        totals[sidx[stratum(r)]] += 1
        members[cluster_of[r.uid]].append(r)

    fracs = np.array(cfg.split_fracs, dtype=np.float64)
    fracs = fracs / fracs.sum()
    target = np.outer(fracs, totals)                         # (splits, strata)
    current = np.zeros_like(target)

    rng = np.random.default_rng(cfg.seed)
    cids = sorted(members)
    rng.shuffle(cids)
    cids.sort(key=lambda c: -len(members[c]))                # stable: random within a size

    split_of: dict[int, str] = {}
    for cid in cids:
        add = np.zeros(len(strata))
        for r in members[cid]:
            add[sidx[stratum(r)]] += 1
        touched = np.flatnonzero(add > 0)
        best_k, best_key = 0, None
        for k in range(len(SPLITS)):
            fill = (current[k, touched] + add[touched]) / np.maximum(target[k, touched], 1e-9)
            key = (float(fill.max()), float(fill.sum()))
            if best_key is None or key < best_key:
                best_k, best_key = k, key
        current[best_k] += add
        for r in members[cid]:
            split_of[r.uid] = SPLITS[best_k]
    return split_of


def _from_manifest(path: str, survivors: list[Record]) -> dict[int, str]:
    """Reuse another run's assignment so a second stage trains on the identical split."""
    with open(path, newline="", encoding="utf-8") as handle:
        lookup = {(row["class"], row["filename"]): row["split"] for row in csv.DictReader(handle)}
    split_of: dict[int, str] = {}
    missing = []
    for r in survivors:
        split = lookup.get((r.cls, r.filename))
        if split is None:
            missing.append(r.filename)
        else:
            split_of[r.uid] = split
    if missing:
        raise RuntimeError(
            f"--split-from {path} has no row for {len(missing)} surviving image(s), e.g. "
            f"{missing[:3]}. The hygiene outcome differs from that run; refusing to mix splits."
        )
    return split_of


def _table(records: list[Record], key) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {s: dict.fromkeys(CLASSES, 0) for s in SPLITS}
    for r in records:
        table[key(r)][r.cls] += 1
    return table


def build_split(
    survivors: list[Record],
    all_records: list[Record],
    clusters: ClusterResult,
    cluster_cover: list[int],
    cfg: TrainConfig,
) -> SplitResult:
    if cfg.split_from:
        split_of = _from_manifest(cfg.split_from, survivors)
        logger.info("split reused from %s", cfg.split_from)
    else:
        split_of = _greedy_assign(survivors, clusters.cluster_of, cfg)
    assert_cluster_purity(clusters.cluster_of, split_of)

    by_split: dict[str, list[Record]] = {s: [] for s in SPLITS}
    for r in survivors:
        by_split[split_of[r.uid]].append(r)

    shipped_of = {r.uid: r.shipped_split for r in all_records}
    cover = set(cluster_cover)
    before = cross_split_counts(clusters, cover, shipped_of)
    after = cross_split_counts(clusters, {r.uid for r in survivors}, split_of)
    if (after["clusters_spanning_splits"] or after["exact_pixel_pairs"]
            or after["phash_confirmed_pairs"] or after["subject_groups_spanning_splits"]):
        raise AssertionError(f"cross-split duplicates survive the re-grouping: {after}")

    moved = Counter((r.shipped_split, split_of[r.uid]) for r in survivors
                    if r.shipped_split != split_of[r.uid])
    n = len(survivors)
    after_counts = _table(survivors, lambda r: split_of[r.uid])
    infection_after = _table([r for r in survivors if r.in_infection_tree], lambda r: split_of[r.uid])
    supervised_after = {s: sum(1 for r in by_split[s] if r.opacity_supervised) for s in SPLITS}
    size_dist = Counter(len(m) for m in clusters.members.values() if any(u in split_of for u in m))
    in_surviving = Counter(sum(1 for u in m if u in split_of) for m in clusters.members.values())
    in_surviving.pop(0, None)

    report: dict[str, Any] = {
        "seed": cfg.seed,
        "reused_from": cfg.split_from,
        "target_fractions": dict(zip(SPLITS, (f / sum(cfg.split_fracs) for f in cfg.split_fracs), strict=True)),
        "achieved_fractions": {s: len(by_split[s]) / max(1, n) for s in SPLITS},
        "images_before_hygiene": len(all_records),
        "images_after_hygiene": n,
        "counts_before_shipped": _table(all_records, lambda r: r.shipped_split),
        "counts_after": after_counts,
        "infection_tree_after": infection_after,
        "opacity_supervised_after": supervised_after,
        "cluster_size_distribution_all": {str(k): v for k, v in sorted(size_dist.items())},
        "cluster_size_distribution_surviving": {str(k): v for k, v in sorted(in_surviving.items())},
        "images_moved": int(sum(moved.values())),
        "images_moved_fraction": float(sum(moved.values()) / max(1, n)),
        "moved_matrix": {f"{a}->{b}": v for (a, b), v in sorted(moved.items())},
        "cross_split_before": before,
        "cross_split_after": after,
        "purity_asserted": True,
    }
    logger.info("split: %s | moved %d (%.1f%%) | cross-split pairs before %s after %s",
                {s: len(by_split[s]) for s in SPLITS}, sum(moved.values()),
                100 * sum(moved.values()) / max(1, n), before, after)
    return SplitResult(split_of=split_of, records=by_split, cluster_of=clusters.cluster_of,
                       report=report)


def write_manifest(result: SplitResult, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sizes = Counter(result.cluster_of[uid] for uid in result.split_of)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_COLUMNS)
        for split in SPLITS:
            for r in result.records[split]:
                cid = result.cluster_of[r.uid]
                writer.writerow([
                    r.uid, r.filename, r.image_path, r.lung_mask_path, r.infection_mask_path or "",
                    r.cls, r.tree, int(r.in_infection_tree), int(r.opacity_supervised), cid,
                    sizes[cid], split, r.shipped_split,
                ])


def load_manifest(path: str) -> SplitResult:
    """Rebuild records and split from ``split_manifest.csv`` alone.

    Used by every training worker and by the resume path, so a restart never needs the
    hygiene pass again and every process sees the identical partition.
    """
    records: dict[str, list[Record]] = {s: [] for s in SPLITS}
    split_of: dict[int, str] = {}
    cluster_of: dict[int, int] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            in_tree = bool(int(row["in_infection_tree"]))
            record = Record(
                uid=int(row["uid"]), filename=row["filename"], cls=row["class"],
                shipped_split=row["shipped_split"], image_path=row["image_path"],
                lung_mask_path=row["lung_mask_path"], in_infection_tree=in_tree,
                infection_mask_path=row["infection_mask_path"] or None,
            )
            if bool(int(row["opacity_supervised"])) != record.opacity_supervised:
                raise RuntimeError(
                    f"Manifest row {row['filename']} carries supervision flag "
                    f"{row['opacity_supervised']} but class {row['class']} / tree {row['tree']} "
                    "implies the opposite; the manifest was edited or is from an older rule."
                )
            records[row["split"]].append(record)
            split_of[record.uid] = row["split"]
            cluster_of[record.uid] = int(row["cluster_id"])
    return SplitResult(split_of=split_of, records=records, cluster_of=cluster_of, report={})


def write_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
        handle.write("\n")
