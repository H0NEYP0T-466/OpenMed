"""Leakage and duplicate hygiene for the chest X-ray dataset.

Pure Pillow + numpy (+ ``imagehash`` when installed). No torch, no CUDA, so
``train_lungs_v2s.py --audit-only`` runs on a CPU laptop in a few minutes and
produces the same report Kaggle produces before a single epoch is spent.

Why this exists: the COVID-19 Radiography Database is assembled from several
public corpora and contains byte-identical and re-encoded duplicates that cross
class boundaries. A random split therefore leaks the same patient into train and
test and inflates macro-F1 by several points. The fix is to cluster images that
are the same picture (identical SHA-256, or pHash within a small Hamming radius)
with union-find and to split by cluster, never by file.

Report sections (``report.json`` / ``report.csv`` / ``report.md``):
    exact duplicates, near duplicates, cross-class conflicts, unreadable files,
    size/mode histograms, class counts and imbalance, leakage audit, verdict.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from PIL import Image

try:  # package import: app.organs.lungs.classification.kaggle.hygiene
    from .config import CLASS_NAMES, list_class_images, source_pool_for
except ImportError:  # direct execution: python train_lungs_v2s.py
    from config import CLASS_NAMES, list_class_images, source_pool_for

try:  # imagehash is the reference implementation; the fallback keeps audit-only usable
    import imagehash

    HAS_IMAGEHASH = True
except ImportError:  # pragma: no cover - exercised only on machines without imagehash
    imagehash = None  # type: ignore[assignment]
    HAS_IMAGEHASH = False

PHASH_BITS = 64
_PHASH_SIZE = 8
_PHASH_HIGHFREQ = 4  # 8 * 4 = 32 px working resolution, same as imagehash.phash


# ─────────────────────────────────────────────────────────────────────────────
# Per-image records
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ImageRecord:
    """One image file plus everything hygiene learned about it."""

    path: str
    rel_path: str
    class_name: str
    class_index: int
    sha256: Optional[str] = None
    phash: Optional[int] = None
    width: int = 0
    height: int = 0
    mode: str = ""
    readable: bool = True
    error: str = ""
    cluster_id: int = -1
    source: str = ""

    @property
    def phash_hex(self) -> Optional[str]:
        return None if self.phash is None else f"{self.phash:016x}"


# ─────────────────────────────────────────────────────────────────────────────
# Hashing
# ─────────────────────────────────────────────────────────────────────────────


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 so a 20 MB X-ray never lands in memory twice."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _dct_matrix(size: int) -> np.ndarray:
    """Orthonormal DCT-II matrix: ``D @ x @ D.T`` is the 2-D transform."""
    indices = np.arange(size)
    matrix = np.cos(np.pi * (2 * indices[None, :] + 1) * indices[:, None] / (2 * size))
    matrix *= np.sqrt(2.0 / size)
    matrix[0, :] *= 1.0 / np.sqrt(2.0)
    return matrix


_DCT_CACHE: dict[int, np.ndarray] = {}


def _phash_fallback(image: Image.Image) -> int:
    """imagehash-compatible 64-bit pHash without the imagehash dependency."""
    size = _PHASH_SIZE * _PHASH_HIGHFREQ
    small = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.float64)
    matrix = _DCT_CACHE.get(size)
    if matrix is None:
        matrix = _dct_matrix(size)
        _DCT_CACHE[size] = matrix
    dct = matrix @ pixels @ matrix.T
    low = dct[:_PHASH_SIZE, :_PHASH_SIZE]
    median = float(np.median(low))
    bits = (low > median).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def phash_of_image(image: Image.Image) -> int:
    """64-bit perceptual hash. ``imagehash.phash`` when available."""
    if HAS_IMAGEHASH:
        return int(str(imagehash.phash(image.convert("L"))), 16)
    return _phash_fallback(image)


def phash_file(path: str | Path) -> int:
    with Image.open(path) as image:
        return phash_of_image(image)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ─────────────────────────────────────────────────────────────────────────────
# Union-find
# ─────────────────────────────────────────────────────────────────────────────


class UnionFind:
    """Disjoint sets with path compression and union by rank."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size
        self.components = size

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, left: int, right: int) -> bool:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return False
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1
        self.components -= 1
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Scanning
# ─────────────────────────────────────────────────────────────────────────────


def _hash_task(task: tuple[str, str, int, str]) -> ImageRecord:
    path, class_name, class_index, rel_path = task
    record = ImageRecord(
        path=path, rel_path=rel_path, class_name=class_name, class_index=class_index
    )
    try:
        record.sha256 = sha256_file(path)
        with Image.open(path) as image:
            record.width, record.height = image.size
            record.mode = image.mode
            record.phash = phash_of_image(image)
    except Exception as exc:  # noqa: BLE001 - any decoder failure is a data defect
        record.readable = False
        record.error = f"{type(exc).__name__}: {exc}"
    return record


def scan(
    class_dir_map: dict[str, Path],
    class_names: Sequence[str],
    dataset_root: str | Path,
    workers: int = 8,
    limit_per_class: int = 0,
) -> list[ImageRecord]:
    """Hash every image of every class folder, in parallel, preserving order.

    ``limit_per_class`` exists for smoke runs on a laptop; 0 means everything.
    """
    root = Path(dataset_root)
    tasks: list[tuple[str, str, int, str]] = []
    for class_name in class_names:
        class_index = list(class_names).index(class_name)
        images = list_class_images(class_dir_map[class_name])
        if limit_per_class > 0:
            images = images[:limit_per_class]
        for path in images:
            try:
                rel = os.path.relpath(path, root)
            except ValueError:
                rel = str(path)
            tasks.append((str(path), class_name, class_index, rel))

    if not tasks:
        return []

    workers = max(1, int(workers))
    if workers == 1:
        return [_hash_task(task) for task in tasks]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_hash_task, tasks))


def cluster_records(
    records: list[ImageRecord], hamming_max: int = 5
) -> tuple[dict[int, list[int]], dict[str, int]]:
    """Union-find over (identical SHA-256, or pHash Hamming <= ``hamming_max``).

    Two hashes are unioned when they are the same file, or when their 64-bit
    perceptual hashes differ in at most ``hamming_max`` bits. Candidate pairs
    come from an 8x8-bit band index: with at most 5 differing bits, at least
    three of the eight byte bands are identical, so no true pair is missed
    without an O(n²) sweep over 21k images.

    Returns ``(cluster_id -> member indices, stats)``; ``record.cluster_id`` is
    set in place.
    """
    readable = [index for index, record in enumerate(records) if record.readable]
    union_find = UnionFind(len(records))

    by_sha: dict[str, list[int]] = defaultdict(list)
    for index in readable:
        if records[index].sha256:
            by_sha[records[index].sha256].append(index)
    for members in by_sha.values():
        for other in members[1:]:
            union_find.union(members[0], other)

    # Collapse equal pHashes first so the band index works on distinct values.
    representatives: dict[int, int] = {}
    for index in readable:
        phash = records[index].phash
        if phash is None:
            continue
        first = representatives.get(phash)
        if first is None:
            representatives[phash] = index
        else:
            union_find.union(first, index)

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for phash, index in representatives.items():
        for band in range(PHASH_BITS // 8):
            buckets[(band, (phash >> (8 * band)) & 0xFF)].append(index)

    pairs_compared = 0
    for members in buckets.values():
        if len(members) < 2:
            continue
        for position, left in enumerate(members):
            left_hash = records[left].phash
            for right in members[position + 1 :]:
                if union_find.find(left) == union_find.find(right):
                    continue
                right_hash = records[right].phash
                if left_hash is None or right_hash is None:
                    continue
                pairs_compared += 1
                if hamming(left_hash, right_hash) <= hamming_max:
                    union_find.union(left, right)

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in readable:
        members_by_root[union_find.find(index)].append(index)

    ordered = sorted(members_by_root.values(), key=lambda members: members[0])
    cluster_members: dict[int, list[int]] = {}
    for cluster_id, members in enumerate(ordered):
        cluster_members[cluster_id] = members
        for index in members:
            records[index].cluster_id = cluster_id

    stats = {
        "readable": len(readable),
        "unreadable": len(records) - len(readable),
        "clusters": len(cluster_members),
        "pairs_compared": pairs_compared,
        "largest_cluster": max((len(m) for m in cluster_members.values()), default=0),
    }
    return cluster_members, stats


def assign_sources(records: list[ImageRecord], dataset_root: str | Path) -> None:
    """Attach the provenance pool of each image's class folder."""
    pools = {name: source_pool_for(name, dataset_root) for name in {r.class_name for r in records}}
    for record in records:
        record.source = pools.get(record.class_name, "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────


def _duplicate_groups(records: list[ImageRecord], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        if not record.readable:
            continue
        value = getattr(record, key)
        if value is None:
            continue
        groups[str(value)].append(record)
    out = []
    for value, members in groups.items():
        if len(members) < 2:
            continue
        out.append(
            {
                "key": value,
                "count": len(members),
                "classes": sorted({m.class_name for m in members}),
                "paths": [m.rel_path for m in members],
            }
        )
    out.sort(key=lambda item: (-item["count"], item["key"]))
    return out


def _near_duplicate_pairs(
    records: list[ImageRecord], cluster_members: dict[int, list[int]], hamming_max: int
) -> int:
    pairs = 0
    for members in cluster_members.values():
        if len(members) < 2:
            continue
        for position, left in enumerate(members):
            for right in members[position + 1 :]:
                a, b = records[left], records[right]
                if a.phash is None or b.phash is None:
                    continue
                if a.sha256 == b.sha256:
                    continue
                if hamming(a.phash, b.phash) <= hamming_max:
                    pairs += 1
    return pairs


def _cluster_has_near_pair(
    records: list[ImageRecord], members: list[int], hamming_max: int
) -> bool:
    """True when a cluster holds at least one *re-encoded* (non-identical) pair."""
    for position, left in enumerate(members):
        for right in members[position + 1 :]:
            a, b = records[left], records[right]
            if a.sha256 == b.sha256:
                continue
            if a.phash is None or b.phash is None:
                continue
            if hamming(a.phash, b.phash) <= hamming_max:
                return True
    return False


def _cross_class_conflicts(
    records: list[ImageRecord], cluster_members: dict[int, list[int]]
) -> list[dict[str, Any]]:
    """Clusters that contain more than one label — the same picture, two answers.

    These are never dropped: the audit lists them so a human can decide. The
    split keeps the cluster whole, which means one of the two labels is
    evaluated on an image the model has already seen — the lesser evil compared
    with a silent duplicate across splits.
    """
    conflicts = []
    for cluster_id, members in cluster_members.items():
        classes = sorted({records[index].class_name for index in members})
        if len(classes) < 2:
            continue
        conflicts.append(
            {
                "cluster_id": cluster_id,
                "classes": classes,
                "count": len(members),
                "by_class": {
                    name: [records[index].rel_path for index in members if records[index].class_name == name]
                    for name in classes
                },
            }
        )
    conflicts.sort(key=lambda item: -item["count"])
    return conflicts


def audit_leakage(
    records: list[ImageRecord], split_of: Optional[dict[int, str]] = None
) -> dict[str, Any]:
    """Clusters spanning splits (must be 0) and source-pool overlap between splits."""
    if not split_of:
        return {
            "computed": False,
            "clusters_spanning_splits": None,
            "examples": [],
            "split_counts": {},
            "source_pools": {},
            "source_pools_in_multiple_splits": None,
        }

    cluster_splits: dict[int, set[str]] = defaultdict(set)
    pool_splits: dict[str, set[str]] = defaultdict(set)
    for index, record in enumerate(records):
        split = split_of.get(index)
        if not split:
            continue
        cluster_splits[record.cluster_id].add(split)
        pool_splits[record.source or "unknown"].add(split)

    spanning = sorted(cid for cid, splits in cluster_splits.items() if len(splits) > 1)
    examples = []
    for cluster_id in spanning[:20]:
        members = [records[i] for i, r in enumerate(records) if r.cluster_id == cluster_id]
        examples.append(
            {
                "cluster_id": cluster_id,
                "splits": sorted(cluster_splits[cluster_id]),
                "paths": [m.rel_path for m in members[:10]],
            }
        )

    return {
        "computed": True,
        "clusters_spanning_splits": len(spanning),
        "examples": examples,
        "split_counts": dict(Counter(split_of.values())),
        "source_pools": {pool: sorted(splits) for pool, splits in sorted(pool_splits.items())},
        "source_pools_in_multiple_splits": sum(
            1 for splits in pool_splits.values() if len(splits) > 1
        ),
    }


def _verdict(report: dict[str, Any]) -> tuple[str, list[str]]:
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    level = "PASS"
    reasons: list[str] = []

    def raise_to(candidate: str, reason: str) -> None:
        nonlocal level
        reasons.append(f"{candidate}: {reason}")
        if order[candidate] > order[level]:
            level = candidate

    total = int(report["num_images"])
    readable = int(report["num_readable"])
    if readable == 0:
        raise_to("FAIL", "no readable images")
        return level, reasons

    unreadable = total - readable
    if unreadable:
        fraction = unreadable / max(total, 1)
        raise_to("FAIL" if fraction > 0.02 else "WARN", f"{unreadable} unreadable file(s)")

    empty_classes = [name for name, count in report["class_counts"].items() if count == 0]
    if empty_classes:
        raise_to("FAIL", f"empty class folder(s): {empty_classes}")

    leakage = report["leakage"]
    if leakage.get("computed") and leakage.get("clusters_spanning_splits"):
        raise_to(
            "FAIL",
            f"{leakage['clusters_spanning_splits']} duplicate cluster(s) span two splits",
        )
    if leakage.get("computed"):
        empty_splits = [
            name for name in ("train", "val", "test") if not leakage["split_counts"].get(name)
        ]
        if empty_splits:
            raise_to("FAIL", f"empty split(s): {empty_splits}")

    if report["cross_class_conflicts"]:
        raise_to(
            "WARN",
            f"{len(report['cross_class_conflicts'])} cluster(s) carry two different labels",
        )
    if report["exact_duplicates"]["groups"]:
        raise_to("WARN", f"{report['exact_duplicates']['groups']} exact duplicate group(s)")
    if report["near_duplicates"]["clusters"]:
        raise_to(
            "WARN",
            f"{report['near_duplicates']['clusters']} near-duplicate cluster(s) were grouped",
        )
    ratio = report["imbalance_ratio"]
    if ratio is not None and ratio > 5.0:
        raise_to("WARN", f"class imbalance {ratio:.2f}:1")

    if not reasons:
        reasons.append("PASS: no duplicates across classes, no leakage, all classes present")
    return level, reasons


def build_report(
    records: list[ImageRecord],
    dataset_root: str | Path,
    cluster_members: dict[int, list[int]],
    hamming_max: int = 5,
    split_of: Optional[dict[int, str]] = None,
    cluster_stats: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble the full hygiene report dictionary."""
    readable = [record for record in records if record.readable]
    class_counts = dict.fromkeys(CLASS_NAMES, 0)
    for record in readable:
        class_counts[record.class_name] = class_counts.get(record.class_name, 0) + 1
    non_zero = [count for count in class_counts.values() if count > 0]
    imbalance = (max(non_zero) / min(non_zero)) if non_zero and min(non_zero) > 0 else None

    sizes = Counter(f"{r.width}x{r.height}" for r in readable if r.width and r.height)
    modes = Counter(r.mode for r in readable if r.mode)

    exact_groups = _duplicate_groups(readable, "sha256")
    near_clusters = [
        {
            "cluster_id": cluster_id,
            "count": len(members),
            "paths": [records[index].rel_path for index in members[:10]],
        }
        for cluster_id, members in sorted(cluster_members.items())
        if len(members) > 1
        and _cluster_has_near_pair(records, members, hamming_max)
    ]

    report: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "hamming_max": int(hamming_max),
        "phash_implementation": "imagehash" if HAS_IMAGEHASH else "numpy-dct-fallback",
        "num_images": len(records),
        "num_readable": len(readable),
        "num_unreadable": len(records) - len(readable),
        "class_counts": class_counts,
        "class_counts_ordered": [class_counts[name] for name in CLASS_NAMES],
        "imbalance_ratio": round(imbalance, 4) if imbalance else None,
        "exact_duplicates": {
            "groups": len(exact_groups),
            "images": sum(group["count"] for group in exact_groups),
            "clusters": exact_groups[:50],
        },
        "near_duplicates": {
            "clusters": len(near_clusters),
            "images": sum(item["count"] for item in near_clusters),
            "pairs": _near_duplicate_pairs(records, cluster_members, hamming_max),
            "examples": near_clusters[:50],
        },
        "cross_class_conflicts": _cross_class_conflicts(records, cluster_members),
        "unreadable": [
            {"path": record.rel_path, "error": record.error}
            for record in records
            if not record.readable
        ],
        "size_mode_histogram": {
            "sizes": dict(sizes.most_common()),
            "modes": dict(modes.most_common()),
        },
        "clusters": {
            "count": len(cluster_members),
            "largest": max((len(m) for m in cluster_members.values()), default=0),
            "histogram": {
                str(size): count
                for size, count in sorted(
                    Counter(len(members) for members in cluster_members.values()).items()
                )
            },
            "stats": cluster_stats or {},
        },
        "leakage": audit_leakage(records, split_of),
        "splits": dict(Counter(split_of.values())) if split_of else {},
    }
    verdict, reasons = _verdict(report)
    report["verdict"] = verdict
    report["verdict_reasons"] = reasons
    return report


def write_report(report: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    """Write ``report.json``, ``report.csv`` and ``report.md`` under ``out_dir``."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    json_path = target / "report.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    md_path = target / "report.md"
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(report))

    return {"json": str(json_path), "md": str(md_path)}


def write_records_csv(
    records: Iterable[ImageRecord],
    path: str | Path,
    split_of: Optional[dict[int, str]] = None,
) -> str:
    """Per-image table: path, class, hashes, cluster, geometry, split."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rel_path",
                "class",
                "sha256",
                "phash",
                "cluster_id",
                "width",
                "height",
                "mode",
                "readable",
                "source",
                "split",
            ]
        )
        for index, record in enumerate(records):
            writer.writerow(
                [
                    record.rel_path,
                    record.class_name,
                    record.sha256 or "",
                    record.phash_hex or "",
                    record.cluster_id,
                    record.width,
                    record.height,
                    record.mode,
                    int(record.readable),
                    record.source,
                    (split_of or {}).get(index, ""),
                ]
            )
    return str(target)


def render_markdown(report: dict[str, Any]) -> str:
    """Human-readable summary, one section per contract item."""
    lines: list[str] = []
    lines.append("# Lung dataset hygiene report")
    lines.append("")
    lines.append(f"- Generated: `{report['generated_at']}`")
    lines.append(f"- Dataset root: `{report['dataset_root']}`")
    lines.append(f"- Images: **{report['num_images']}** ({report['num_readable']} readable)")
    lines.append(f"- pHash: `{report['phash_implementation']}`, Hamming ≤ {report['hamming_max']}")
    lines.append(f"- **Verdict: `{report['verdict']}`**")
    lines.append("")
    lines.append("## Class counts")
    lines.append("")
    lines.append("| Class | Images |")
    lines.append("|---|---|")
    for name, count in report["class_counts"].items():
        lines.append(f"| {name} | {count} |")
    ratio = report["imbalance_ratio"]
    lines.append("")
    lines.append(f"Imbalance ratio: **{ratio if ratio is not None else 'n/a'}**")
    lines.append("")
    lines.append("## Duplicates")
    lines.append("")
    exact = report["exact_duplicates"]
    near = report["near_duplicates"]
    lines.append(f"- Exact duplicates (SHA-256): {exact['groups']} group(s), {exact['images']} file(s)")
    lines.append(
        f"- Near duplicates (pHash): {near['clusters']} cluster(s), {near['images']} file(s), "
        f"{near['pairs']} pair(s)"
    )
    lines.append(f"- Clusters total: {report['clusters']['count']} (largest {report['clusters']['largest']})")
    lines.append("")
    lines.append("## Cross-class conflicts")
    lines.append("")
    conflicts = report["cross_class_conflicts"]
    if conflicts:
        lines.append(f"{len(conflicts)} cluster(s) contain more than one label. Listed, never dropped.")
        lines.append("")
        for item in conflicts[:20]:
            lines.append(
                f"- cluster {item['cluster_id']}: {', '.join(item['classes'])} "
                f"({item['count']} files)"
            )
    else:
        lines.append("None.")
    lines.append("")
    lines.append("## Unreadable files")
    lines.append("")
    unreadable = report["unreadable"]
    if unreadable:
        for item in unreadable[:50]:
            lines.append(f"- `{item['path']}` — {item['error']}")
    else:
        lines.append("None.")
    lines.append("")
    lines.append("## Size and mode histogram")
    lines.append("")
    lines.append("| Size | Count |")
    lines.append("|---|---|")
    for size, count in report["size_mode_histogram"]["sizes"].items():
        lines.append(f"| {size} | {count} |")
    lines.append("")
    lines.append("Modes: " + ", ".join(f"{k}x{v}" for k, v in report["size_mode_histogram"]["modes"].items()))
    lines.append("")
    lines.append("## Leakage audit")
    lines.append("")
    leakage = report["leakage"]
    if leakage.get("computed"):
        lines.append(f"- Clusters spanning splits: **{leakage['clusters_spanning_splits']}** (must be 0)")
        lines.append(f"- Split sizes: {leakage['split_counts']}")
        lines.append(
            "- Source pools in more than one split: "
            f"{leakage['source_pools_in_multiple_splits']}"
        )
        lines.append("")
        lines.append("| Source pool | Splits |")
        lines.append("|---|---|")
        for pool, splits in leakage["source_pools"].items():
            lines.append(f"| {pool} | {', '.join(splits)} |")
    else:
        lines.append("Split not assigned yet; run without `--audit-only` to compute it.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"`{report['verdict']}`")
    lines.append("")
    for reason in report["verdict_reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines)


def summary_lines(report: dict[str, Any]) -> list[str]:
    """Compact console summary, one line per report section."""
    return [
        f"images={report['num_images']} readable={report['num_readable']} "
        f"unreadable={report['num_unreadable']}",
        f"classes={report['class_counts']} imbalance={report['imbalance_ratio']}",
        f"exact_duplicate_groups={report['exact_duplicates']['groups']} "
        f"near_duplicate_clusters={report['near_duplicates']['clusters']} "
        f"clusters={report['clusters']['count']}",
        f"cross_class_conflicts={len(report['cross_class_conflicts'])}",
        f"leakage={report['leakage'].get('clusters_spanning_splits')} "
        f"splits={report['leakage'].get('split_counts')}",
        f"VERDICT={report['verdict']}",
    ]


def run_audit(
    class_dir_map: dict[str, Path],
    class_names: Sequence[str],
    dataset_root: str | Path,
    out_dir: str | Path,
    workers: int = 8,
    hamming_max: int = 5,
    split_of: Optional[dict[int, str]] = None,
    limit_per_class: int = 0,
) -> tuple[list[ImageRecord], dict[int, list[int]], dict[str, Any]]:
    """Scan → cluster → report → write. Returns the records so callers can reuse them."""
    records = scan(class_dir_map, class_names, dataset_root, workers, limit_per_class)
    assign_sources(records, dataset_root)
    cluster_members, cluster_stats = cluster_records(records, hamming_max)
    report = build_report(
        records, dataset_root, cluster_members, hamming_max, split_of, cluster_stats
    )
    write_report(report, out_dir)
    write_records_csv(records, Path(out_dir) / "report.csv", split_of)
    return records, cluster_members, report


# ─────────────────────────────────────────────────────────────────────────────
# Conflict resolution and cleaning
# ─────────────────────────────────────────────────────────────────────────────


def _infer_class(path: str, known_classes: Sequence[str] = CLASS_NAMES) -> str:
    """Extract or match the class folder from an image path."""
    parts = Path(path).parts
    if parts and parts[0] in known_classes:
        return parts[0]
    for cls in known_classes:
        if path.startswith(cls + "/") or path.startswith(cls + "\\"):
            return cls
    return ""


def render_conflict_resolution_markdown(
    dropped_exact_count: int,
    relabeled_count: int,
    dropped_tie_count: int,
    dropped_near_single_count: int,
    kept_count: int,
    imbalance_ratio: Optional[float],
) -> str:
    """Markdown section summarizing the conflict resolution results."""
    ratio_str = f"{imbalance_ratio:.4f}:1" if imbalance_ratio is not None else "n/a"
    lines = [
        "## Conflict resolution",
        "",
        f"- Exact duplicates dropped: **{dropped_exact_count}**",
        f"- Cross-class conflicts relabeled (majority): **{relabeled_count}**",
        f"- Cross-class conflict ties dropped: **{dropped_tie_count}**",
        f"- Single-label near-duplicates collapsed: **{dropped_near_single_count}**",
        f"- Kept images: **{kept_count}**",
        f"- New imbalance ratio: **{ratio_str}**",
    ]
    return "\n".join(lines)


def resolve_conflicts(
    report_json_path: str | Path,
    output_dir: str | Path,
    collapse_single_label: bool = False,
    apply: bool = False,
    dataset_root: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Clean exact duplicates, cross-class conflicts, and optional near-duplicates.

    Reads ``report.json`` produced by ``run_audit`` / ``train_lungs_v2s.py --audit-only``
    and executes three sequential operations:
    1. Exact dedup: For every group in exact_duplicates, keeps the first path,
       drops the rest as dropped_exact.
    2. Consensus relabel: For every entry in cross_class_conflicts, picks the majority
       class (most files). For files whose folder label != majority, records them as
       relabeled with old_label/new_label. Ties drop the entire cluster under dropped_tie.
    3. Optional single-label near-dup collapse (when collapse_single_label=True):
       For every near-duplicate cluster whose files are all one class, keeps one representative
       path, drops the rest as dropped_near_single. Clusters handled in step 2 are skipped.

    Writes ``cleaned_manifest.json`` and appends a "## Conflict resolution" section to
    ``report.md`` under ``output_dir``. If ``apply=True``, physically moves dropped files
    to ``<output_dir>/dropped/`` and moves relabeled files to their new class folder.
    """
    report_path = Path(report_json_path)
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    # 1. Exact dedup
    exact_data = report.get("exact_duplicates", {})
    if isinstance(exact_data, list):
        exact_groups = exact_data
    elif isinstance(exact_data, dict):
        exact_groups = exact_data.get("clusters") or exact_data.get("groups") or []
    else:
        exact_groups = []

    dropped_exact: list[str] = []
    dropped_exact_set: set[str] = set()

    for group in exact_groups:
        paths = group if isinstance(group, list) else group.get("paths", [])
        if len(paths) > 1:
            for p in paths[1:]:
                dropped_exact.append(p)
                dropped_exact_set.add(p)

    # 2. Consensus relabel
    conflicts = report.get("cross_class_conflicts", [])
    dropped_tie: list[str] = []
    dropped_tie_set: set[str] = set()
    relabeled: list[dict[str, str]] = []
    handled_clusters: set[int] = set()

    for conf in conflicts:
        cid = conf.get("cluster_id")
        if cid is not None:
            handled_clusters.add(cid)
        by_class = conf.get("by_class", {})
        if not by_class:
            continue

        counts = {cls: len(paths) for cls, paths in by_class.items()}
        max_c = max(counts.values()) if counts else 0
        top_classes = [cls for cls, c in counts.items() if c == max_c]

        if len(top_classes) > 1:
            for paths in by_class.values():
                for p in paths:
                    if p not in dropped_exact_set and p not in dropped_tie_set:
                        dropped_tie.append(p)
                        dropped_tie_set.add(p)
        else:
            majority_class = top_classes[0]
            for old_cls, paths in by_class.items():
                if old_cls != majority_class:
                    for p in paths:
                        if p not in dropped_exact_set:
                            relabeled.append({
                                "path": p,
                                "old_label": old_cls,
                                "new_label": majority_class,
                            })

    # 3. Optional single-label near-dup collapse
    dropped_near_single: list[str] = []
    dropped_near_set: set[str] = set()

    if collapse_single_label:
        near_data = report.get("near_duplicates", {})
        if isinstance(near_data, list):
            near_groups = near_data
        elif isinstance(near_data, dict):
            near_groups = (
                near_data.get("clusters")
                if isinstance(near_data.get("clusters"), list)
                else near_data.get("examples", [])
            )
        else:
            near_groups = []

        for group in near_groups:
            cid = group.get("cluster_id") if isinstance(group, dict) else None
            if cid is not None and cid in handled_clusters:
                continue
            paths = group if isinstance(group, list) else group.get("paths", [])
            if not paths:
                continue

            classes_in_group = set()
            for p in paths:
                cls = _infer_class(p, CLASS_NAMES)
                if cls:
                    classes_in_group.add(cls)
            if isinstance(group, dict) and "classes" in group and group["classes"]:
                classes_in_group.update(group["classes"])

            if len(classes_in_group) == 1:
                available = [
                    p
                    for p in paths
                    if p not in dropped_exact_set and p not in dropped_tie_set
                ]
                if len(available) > 1:
                    for p in available[1:]:
                        if p not in dropped_near_set:
                            dropped_near_single.append(p)
                            dropped_near_set.add(p)

    # Class counts recomputation
    initial_counts = report.get("class_counts", {})
    class_counts_after = {name: int(initial_counts.get(name, 0)) for name in CLASS_NAMES}

    for p in dropped_exact:
        cls = _infer_class(p, CLASS_NAMES)
        if cls in class_counts_after:
            class_counts_after[cls] -= 1

    for p in dropped_tie:
        cls = _infer_class(p, CLASS_NAMES)
        if cls in class_counts_after:
            class_counts_after[cls] -= 1

    for p in dropped_near_single:
        cls = _infer_class(p, CLASS_NAMES)
        if cls in class_counts_after:
            class_counts_after[cls] -= 1

    for item in relabeled:
        old_lbl = item["old_label"]
        new_lbl = item["new_label"]
        if old_lbl in class_counts_after:
            class_counts_after[old_lbl] -= 1
        if new_lbl in class_counts_after:
            class_counts_after[new_lbl] += 1

    total_dropped = len(dropped_exact) + len(dropped_tie) + len(dropped_near_single)
    num_images = int(report.get("num_images", sum(initial_counts.values())))
    kept_count = num_images - total_dropped

    non_zero = [c for c in class_counts_after.values() if c > 0]
    new_imbalance = (max(non_zero) / min(non_zero)) if non_zero and min(non_zero) > 0 else None

    # Write cleaned_manifest.json
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "cleaned_manifest.json"
    manifest: dict[str, Any] = {
        "kept_count": kept_count,
        "dropped_exact": dropped_exact,
        "relabeled": relabeled,
        "dropped_tie": dropped_tie,
        "dropped_near_single": dropped_near_single,
        "class_counts_after": class_counts_after,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    # Update or append report.md
    md_section = render_conflict_resolution_markdown(
        dropped_exact_count=len(dropped_exact),
        relabeled_count=len(relabeled),
        dropped_tie_count=len(dropped_tie),
        dropped_near_single_count=len(dropped_near_single),
        kept_count=kept_count,
        imbalance_ratio=new_imbalance,
    )
    target_md = out_dir / "report.md"
    if not target_md.is_file() and (report_path.parent / "report.md").is_file():
        target_md = report_path.parent / "report.md"

    if target_md.is_file():
        existing = target_md.read_text(encoding="utf-8")
        if "## Conflict resolution" in existing:
            parts = existing.split("## Conflict resolution")
            head = parts[0].rstrip()
            rest = parts[1]
            next_header_idx = -1
            for line_idx, line in enumerate(rest.splitlines()):
                if line.startswith("## ") and line_idx > 0:
                    next_header_idx = rest.find(line)
                    break
            tail = rest[next_header_idx:] if next_header_idx != -1 else ""
            target_md.write_text(f"{head}\n\n{md_section}\n{tail}".strip() + "\n", encoding="utf-8")
        else:
            with open(target_md, "a", encoding="utf-8") as handle:
                handle.write("\n\n" + md_section + "\n")
    else:
        target_md.write_text(md_section + "\n", encoding="utf-8")

    # Optional physical moves if apply=True
    if apply:
        root = Path(dataset_root or report.get("dataset_root", "."))
        dropped_dir = out_dir / "dropped"
        all_dropped = dropped_exact + dropped_tie + dropped_near_single
        for p in all_dropped:
            src = root / p if not Path(p).is_absolute() else Path(p)
            if src.is_file():
                dst = dropped_dir / p
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

        for item in relabeled:
            p = item["path"]
            old_label = item["old_label"]
            new_label = item["new_label"]
            src = root / p if not Path(p).is_absolute() else Path(p)
            if src.is_file():
                parts = list(Path(p).parts)
                if parts and parts[0] == old_label:
                    parts[0] = new_label
                    new_rel = Path(*parts)
                else:
                    new_rel = Path(p.replace(old_label, new_label, 1))
                dst = root / new_rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

    return manifest


__all__ = [
    "HAS_IMAGEHASH",
    "ImageRecord",
    "UnionFind",
    "assign_sources",
    "audit_leakage",
    "build_report",
    "cluster_records",
    "hamming",
    "phash_file",
    "phash_of_image",
    "render_conflict_resolution_markdown",
    "render_markdown",
    "resolve_conflicts",
    "run_audit",
    "scan",
    "sha256_file",
    "summary_lines",
    "write_records_csv",
    "write_report",
]
