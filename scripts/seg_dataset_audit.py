"""
Segmentation dataset hygiene audit.
Dataset: /home/honeypot/Downloads/archive/{images,masks}

Checks: pair integrity, dimensions, mask encoding, degenerate masks,
SHA256 exact duplicates (within + across), perceptual-hash near duplicates
(same-series vs cross-series), series/group structure for leakage-safe splits,
and class/sequence distribution parsed from filenames.

Read-only. Writes nothing except the report to stdout.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = "/home/honeypot/Downloads/archive"
IMG_DIR = os.path.join(ROOT, "images")
MSK_DIR = os.path.join(ROOT, "masks")

PHASH_THRESHOLD = 6  # Hamming distance for "near duplicate"
SEQ_RE = re.compile(r"^(T1C\+|T1|T2)\s*-\s*")
TRAIL_NUM_RE = re.compile(r"\s+\d+$")


def series_key(stem: str) -> str:
    """'T1C+ - Meningioma tentorial , ventricle 038' -> 'T1C+ - Meningioma tentorial , ventricle'"""
    return TRAIL_NUM_RE.sub("", stem)


def base_key(stem: str) -> str:
    """Series key with the sequence prefix stripped — the same lesion under any sequence."""
    return SEQ_RE.sub("", series_key(stem))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_bits(path: str, hash_size: int = 8, factor: int = 4) -> np.uint64 | None:
    """DCT-based perceptual hash, returned as a 64-bit int."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    size = hash_size * factor
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:hash_size, :hash_size]
    flat = dct.flatten()
    med = np.median(flat[1:])  # drop DC term
    bits = flat > med
    val = np.uint64(0)
    for b in bits:
        val = (val << np.uint64(1)) | np.uint64(1 if b else 0)
    return val


def popcount64(x: np.ndarray) -> np.ndarray:
    """SWAR popcount for a uint64 array."""
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint8)


def analyse_pair(stem: str) -> dict:
    """Per-pair geometry + mask sanity."""
    ip = os.path.join(IMG_DIR, stem + ".jpg")
    mp = os.path.join(MSK_DIR, stem + ".png")

    img = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
    msk = cv2.imread(mp, cv2.IMREAD_UNCHANGED)

    rec: dict = {"stem": stem, "img_shape": None, "msk_shape": None,
                 "img_bad": img is None, "msk_bad": msk is None,
                 "msk_values": None, "fg_frac": None, "mismatch": False}
    if img is None or msk is None:
        return rec

    rec["img_shape"] = img.shape
    rec["msk_shape"] = msk.shape
    rec["mismatch"] = img.shape[:2] != msk.shape[:2]
    if msk.ndim == 2:
        uniq = np.unique(msk)
        rec["msk_values"] = uniq[:8].tolist()
        rec["n_values"] = int(uniq.size)
        rec["fg_frac"] = float((msk > 0).mean())
    return rec


def main() -> None:
    img_files = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    msk_files = sorted(f for f in os.listdir(MSK_DIR) if f.lower().endswith(".png"))

    img_stems = {os.path.splitext(f)[0] for f in img_files}
    msk_stems = {os.path.splitext(f)[0] for f in msk_files}

    print("=" * 78)
    print("1. PAIR INTEGRITY")
    print("=" * 78)
    print(f"  images: {len(img_files)}   masks: {len(msk_files)}")
    print(f"  image-without-mask: {len(img_stems - msk_stems)}")
    print(f"  mask-without-image: {len(msk_stems - img_stems)}")

    stems = sorted(img_stems & msk_stems)
    print(f"  matched pairs: {len(stems)}")

    # ── per-pair geometry + mask sanity (parallel) ────────────────────────
    print()
    print("=" * 78)
    print("2. GEOMETRY & MASK ENCODING")
    print("=" * 78)
    with Pool(os.cpu_count()) as pool:
        recs = pool.map(analyse_pair, stems, chunksize=64)

    bad_img = [r["stem"] for r in recs if r["img_bad"]]
    bad_msk = [r["stem"] for r in recs if r["msk_bad"]]
    mismatched = [r["stem"] for r in recs if r["mismatch"]]

    img_shapes = Counter(str(r["img_shape"]) for r in recs if r["img_shape"])
    msk_shapes = Counter(str(r["msk_shape"]) for r in recs if r["msk_shape"])
    print(f"  unreadable images: {len(bad_img)}   unreadable masks: {len(bad_msk)}")
    print(f"  image/mask dimension mismatches: {len(mismatched)}")
    print(f"  distinct image shapes: {dict(img_shapes)}")
    print(f"  distinct mask shapes:  {dict(msk_shapes)}")

    val_sets = Counter(tuple(r["msk_values"]) for r in recs if r["msk_values"])
    print(f"  distinct mask value-sets: {dict(val_sets)}")

    fg = np.array([r["fg_frac"] for r in recs if r["fg_frac"] is not None])
    empty = [r["stem"] for r in recs if r["fg_frac"] == 0.0]
    full = [r["stem"] for r in recs if r["fg_frac"] == 1.0]
    print(f"  foreground fraction: min={fg.min():.4f}  p25={np.percentile(fg,25):.4f}  "
          f"median={np.median(fg):.4f}  p75={np.percentile(fg,75):.4f}  max={fg.max():.4f}")
    print(f"  EMPTY masks (all background): {len(empty)}")
    print(f"  FULL masks (all foreground): {len(full)}")
    if empty:
        print(f"    e.g. {empty[:5]}")
    tiny = [r["stem"] for r in recs if r["fg_frac"] is not None and 0 < r["fg_frac"] < 0.001]
    print(f"  tiny masks (<0.1% fg, <262 px): {len(tiny)}")

    # ── SHA256 exact duplicates ───────────────────────────────────────────
    print()
    print("=" * 78)
    print("3. SHA256 EXACT DUPLICATES")
    print("=" * 78)
    with Pool(os.cpu_count()) as pool:
        img_sha = pool.map(sha256, [os.path.join(IMG_DIR, s + ".jpg") for s in stems], chunksize=64)
        msk_sha = pool.map(sha256, [os.path.join(MSK_DIR, s + ".png") for s in stems], chunksize=64)

    def dup_groups(hashes: list[str]) -> dict[str, list[int]]:
        seen: dict[str, list[int]] = defaultdict(list)
        for i, h in enumerate(hashes):
            seen[h].append(i)
        return {h: idx for h, idx in seen.items() if len(idx) > 1}

    img_dups = dup_groups(img_sha)
    msk_dups = dup_groups(msk_sha)
    print(f"  image files: {len(stems)}   unique image hashes: {len(set(img_sha))}")
    print(f"  mask files:  {len(stems)}   unique mask hashes:  {len(set(msk_sha))}")
    print(f"  duplicate IMAGE groups: {len(img_dups)}  "
          f"(extra files: {sum(len(v) - 1 for v in img_dups.values())})")
    print(f"  duplicate MASK groups:  {len(msk_dups)}  "
          f"(extra files: {sum(len(v) - 1 for v in msk_dups.values())})")

    img_hash_set = set(img_sha)
    msk_hash_set = set(msk_sha)
    cross = img_hash_set & msk_hash_set
    print(f"  image hashes that also appear as a mask (and vice-versa): {len(cross)}")

    for label, groups in (("IMAGE", img_dups), ("MASK", msk_dups)):
        for h, idx in list(groups.items())[:5]:
            names = [stems[i] for i in idx][:4]
            print(f"    {label} dup x{len(idx)}: {names}")

    # ── series / grouping structure ───────────────────────────────────────
    print()
    print("=" * 78)
    print("4. SERIES / GROUP STRUCTURE (for leakage-safe splits)")
    print("=" * 78)
    series: dict[str, list[str]] = defaultdict(list)
    for s in stems:
        series[series_key(s)].append(s)
    bases: dict[str, list[str]] = defaultdict(list)
    for s in stems:
        bases[base_key(s)].append(s)

    sizes = np.array([len(v) for v in series.values()])
    print(f"  unique series (stem minus trailing number): {len(series)}")
    print(f"  slices per series: min={sizes.min()}  median={int(np.median(sizes))}  "
          f"mean={sizes.mean():.1f}  max={sizes.max()}")
    print(f"  unique BASE keys (sequence prefix also stripped): {len(bases)}")
    multi = {k: v for k, v in bases.items() if len({series_key(x) for x in v}) > 1}
    print(f"  base keys spanning >1 sequence series: {len(multi)}")
    for k, v in list(multi.items())[:3]:
        print(f"    {k!r} -> {sorted({series_key(x) for x in v})[:4]}")

    seq_counts = Counter((SEQ_RE.match(s).group(1) if SEQ_RE.match(s) else "(none)") for s in stems)
    print(f"  sequence prefix in filename: {dict(seq_counts)}")

    # ── perceptual hashes ─────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("5. PERCEPTUAL HASH — NEAR DUPLICATES")
    print("=" * 78)
    print(f"  hashing {len(stems)} images (pHash 8x8, DCT) ...")
    sys.stdout.flush()
    with Pool(os.cpu_count()) as pool:
        ph = pool.map(phash_bits, [os.path.join(IMG_DIR, s + ".jpg") for s in stems], chunksize=64)

    ok = [(i, h) for i, h in enumerate(ph) if h is not None]
    idxs = np.array([i for i, _ in ok])
    hashes = np.array([h for _, h in ok], dtype=np.uint64)
    print(f"  hashed OK: {len(hashes)}   failed: {len(ph) - len(hashes)}")

    same_series, cross_series = 0, 0
    examples_same: list[tuple[str, str, int]] = []
    examples_cross: list[tuple[str, str, int]] = []

    CH = 512
    for start in range(0, len(hashes), CH):
        block = hashes[start:start + CH]
        d = popcount64(block[:, None] ^ hashes[None, :])
        rows, cols = np.nonzero(d <= PHASH_THRESHOLD)
        for r, c in zip(rows, cols):
            gi, gj = start + r, c
            if gi >= gj:
                continue
            si, sj = stems[idxs[gi]], stems[idxs[gj]]
            dist = int(d[r, c])
            if series_key(si) == series_key(sj):
                same_series += 1
                if len(examples_same) < 4:
                    examples_same.append((si, sj, dist))
            else:
                cross_series += 1
                if len(examples_cross) < 12:
                    examples_cross.append((si, sj, dist))

    total_pairs = same_series + cross_series
    print(f"  near-duplicate pairs (Hamming <= {PHASH_THRESHOLD}): {total_pairs}")
    print(f"    within the same series (expected — adjacent slices): {same_series}")
    print(f"    ACROSS different series (needs a look):              {cross_series}")

    if examples_cross:
        print("  cross-series examples:")
        for a, b, dd in examples_cross:
            print(f"    d={dd}  {a}")
            print(f"          {b}")

    # cross-series clusters
    if cross_series:
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for start in range(0, len(hashes), CH):
            block = hashes[start:start + CH]
            d = popcount64(block[:, None] ^ hashes[None, :])
            rows, cols = np.nonzero(d <= PHASH_THRESHOLD)
            for r, c in zip(rows, cols):
                gi, gj = start + r, c
                if gi >= gj:
                    continue
                si, sj = stems[idxs[gi]], stems[idxs[gj]]
                if series_key(si) != series_key(sj):
                    union(si, sj)
        clusters: dict[str, list[str]] = defaultdict(list)
        for k in list(parent):
            clusters[find(k)].append(k)
        clusters = {k: v for k, v in clusters.items() if len(v) > 1}
        sizes_c = sorted((len(v) for v in clusters.values()), reverse=True)
        print(f"  cross-series near-dup clusters: {len(clusters)}  "
              f"largest sizes: {sizes_c[:8]}")

    # ── class distribution ────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("6. CLASS DISTRIBUTION (parsed from filename, informational)")
    print("=" * 78)
    cls: Counter = Counter()
    for s in stems:
        t = SEQ_RE.sub("", series_key(s))
        for sep in (" , ", " - ", " (", ","):
            if sep in t:
                t = t.split(sep)[0]
                break
        cls[t.strip()] += 1
    print(f"  distinct leading tumour-type tokens: {len(cls)}")
    for name, n in cls.most_common(15):
        print(f"    {n:6d}  {name}")

    print()
    print("=" * 78)
    print("AUDIT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
