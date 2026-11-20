"""Duplicate clusters: union-find over exact hashes and confirmed perceptual hashes.

Edges
    exact bytes   identical SHA-256 of the raw file
    exact pixels  identical SHA-256 of the decoded pixels (the same image
                  re-encoded has different bytes and the same pixels)
    near          DCT pHash within ``phash_hamming_max`` bits *and* confirmed at the
                  pixel level (mean absolute difference and correlation). A hash
                  collision between two different patients must never glue them
                  into one cluster.

Partitioning clusters, never images, is what stops a split from leaking: a cluster
spanning two splits is a hard assertion failure (``splits.assert_cluster_purity``).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("clusters")

_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(values: np.ndarray) -> np.ndarray:
    """Per-element population count of a ``uint64`` array."""
    values = np.ascontiguousarray(values, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values).astype(np.int64)
    as_bytes = values.view(np.uint8).reshape(*values.shape, 8)
    return _POPCOUNT_TABLE[as_bytes].sum(axis=-1).astype(np.int64)


class UnionFind:
    def __init__(self, items: list[int]) -> None:
        self.parent = {i: i for i in items}
        self.size = dict.fromkeys(items, 1)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def phash64(gray: np.ndarray) -> int:
    """64-bit DCT perceptual hash (32x32 area-resized, 8x8 low-frequency block, median split)."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].reshape(-1)
    bits = low > np.median(low)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


@lru_cache(maxsize=1024)
def _read_gray(path: str) -> np.ndarray:
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"unreadable: {path}")
    return gray


def pixel_agreement(path_a: str, path_b: str) -> tuple[float, float]:
    """``(mean absolute difference, Pearson correlation)`` of two images."""
    a = _read_gray(path_a).astype(np.float32)
    b = _read_gray(path_b).astype(np.float32)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    mae = float(np.abs(a - b).mean())
    sa, sb = float(a.std()), float(b.std())
    corr = 1.0 if (sa == 0 and sb == 0 and mae == 0) else (
        0.0 if sa == 0 or sb == 0 else float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
    )
    return mae, corr


def phash_candidate_pairs(
    hashes: np.ndarray, max_dist: int, bucket_cap: int
) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    """All index pairs within ``max_dist`` bits, found by banding.

    Pigeonhole: split the 64 bits into ``max_dist + 1`` bands; two hashes within
    ``max_dist`` bits must agree exactly on at least one band. Buckets larger than
    ``bucket_cap`` (blank or near-uniform images collide en masse) are skipped and
    counted so the report can say what was not examined.
    """
    n_bands = max_dist + 1
    edges = np.linspace(0, 64, n_bands + 1).astype(int)
    pairs: dict[tuple[int, int], int] = {}
    skipped_buckets = 0
    skipped_items = 0
    for band in range(n_bands):
        lo, hi = int(edges[band]), int(edges[band + 1])
        width = hi - lo
        mask = np.uint64((1 << width) - 1)
        keys = (hashes >> np.uint64(lo)) & mask
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        ends = np.r_[starts[1:], len(order)]
        for start, end in zip(starts, ends, strict=True):
            m = int(end - start)
            if m < 2:
                continue
            if m > bucket_cap:
                skipped_buckets += 1
                skipped_items += m
                continue
            idx = order[start:end]
            xor = hashes[idx][:, None] ^ hashes[idx][None, :]
            dist = popcount64(xor)
            ii, jj = np.nonzero(np.triu(dist <= max_dist, k=1))
            for a, b in zip(ii.tolist(), jj.tolist(), strict=True):
                x, y = int(idx[a]), int(idx[b])
                pairs[(min(x, y), max(x, y))] = int(dist[a, b])
    return pairs, {"bands": n_bands, "skipped_buckets": skipped_buckets,
                   "skipped_items": skipped_items}


@dataclass
class ClusterResult:
    cluster_of: dict[int, int]                     # uid -> cluster id
    members: dict[int, list[int]]                  # cluster id -> uids
    exact_pixel_pairs: list[tuple[int, int]] = field(default_factory=list)
    exact_byte_pairs: list[tuple[int, int]] = field(default_factory=list)
    near_pairs: dict[tuple[int, int], dict[str, float]] = field(default_factory=dict)
    exact_groups: list[list[int]] = field(default_factory=list)   # pixel-identical groups
    subject_of: dict[int, str] = field(default_factory=dict)      # uid -> BIDS subject id
    stats: dict[str, Any] = field(default_factory=dict)

    def size_distribution(self) -> dict[str, int]:
        return dict(sorted(Counter(len(m) for m in self.members.values()).items()))


def _group_pairs(groups: dict[str, list[int]]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for members in groups.values():
        if len(members) > 1:
            pairs.extend(combinations(sorted(members), 2))
    return pairs


def build_clusters(
    uids: list[int],
    sha_bytes: dict[int, str],
    sha_pixels: dict[int, str],
    phashes: dict[int, int],
    paths: dict[int, str],
    *,
    phash_hamming_max: int,
    pixel_mae: float,
    min_corr: float,
    bucket_cap: int,
    subject_keys: dict[int, str] | None = None,
) -> ClusterResult:
    uf = UnionFind(list(uids))
    by_bytes: dict[str, list[int]] = defaultdict(list)
    by_pixels: dict[str, list[int]] = defaultdict(list)
    for uid in uids:
        by_bytes[sha_bytes[uid]].append(uid)
        by_pixels[sha_pixels[uid]].append(uid)

    byte_pairs = _group_pairs(by_bytes)
    pixel_pairs = _group_pairs(by_pixels)
    for a, b in pixel_pairs + byte_pairs:
        uf.union(a, b)

    # Same subject id means the same patient: images of one person must not straddle
    # splits any more than a re-encoded copy may. Only files that carry a subject id
    # (BIDS-style names) take part; ``covid_<n>`` names carry none.
    subject_of = {u: k for u, k in (subject_keys or {}).items() if u in uf.parent}
    by_subject: dict[str, list[int]] = defaultdict(list)
    for uid, key in subject_of.items():
        by_subject[key].append(uid)
    subject_unions = 0
    for members_of_subject in by_subject.values():
        for other in members_of_subject[1:]:
            uf.union(members_of_subject[0], other)
            subject_unions += 1

    near: dict[tuple[int, int], dict[str, float]] = {}
    stats: dict[str, Any] = {"phash_candidates": 0, "phash_confirmed": 0, "phash_rejected": 0,
                             "subject_ids": len(by_subject), "subject_unions": subject_unions,
                             "largest_subject_group": max((len(m) for m in by_subject.values()),
                                                          default=0)}
    if len(uids) > 1 and phash_hamming_max >= 0:
        hashes = np.array([phashes[u] for u in uids], dtype=np.uint64)
        candidates, band_stats = phash_candidate_pairs(hashes, phash_hamming_max, bucket_cap)
        stats.update(band_stats)
        stats["phash_candidates"] = len(candidates)
        exact_set = {tuple(sorted(p)) for p in pixel_pairs}
        for (i, j), dist in candidates.items():
            a, b = uids[i], uids[j]
            key = (min(a, b), max(a, b))
            if key in exact_set:
                continue
            mae, corr = pixel_agreement(paths[a], paths[b])
            if mae <= pixel_mae and corr >= min_corr:
                near[key] = {"hamming": dist, "mae": mae, "corr": corr}
                uf.union(a, b)
            else:
                stats["phash_rejected"] += 1
        stats["phash_confirmed"] = len(near)

    roots: dict[int, list[int]] = defaultdict(list)
    for uid in uids:
        roots[uf.find(uid)].append(uid)
    ordered = sorted(roots.values(), key=lambda m: min(m))
    members = {cid: sorted(m) for cid, m in enumerate(ordered)}
    cluster_of = {uid: cid for cid, m in members.items() for uid in m}
    exact_groups = [sorted(m) for m in by_pixels.values() if len(m) > 1]
    stats["clusters"] = len(members)
    stats["multi_member_clusters"] = sum(1 for m in members.values() if len(m) > 1)
    stats["largest_cluster"] = max((len(m) for m in members.values()), default=0)
    return ClusterResult(
        cluster_of=cluster_of, members=members, exact_pixel_pairs=pixel_pairs,
        exact_byte_pairs=byte_pairs, near_pairs=near, exact_groups=exact_groups,
        subject_of=subject_of, stats=stats,
    )
