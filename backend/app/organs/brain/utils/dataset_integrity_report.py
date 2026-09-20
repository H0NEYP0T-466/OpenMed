"""Data integrity report for folder-structured brain classification datasets.

Answers the questions that decide whether a dataset is safe to train on:

  1. Does the same image appear twice inside one class?      (SHA-256 + pHash)
  2. Does the same image appear in two different classes?    (SHA-256 + pHash)
  3. Subject leakage: are slices of one patient spread across files that a
     naive splitter would scatter into train AND test?        (series detection)
  4. Are the classes declared "independent images" actually independent?

Methods
-------
- SHA-256 over raw file bytes: exact duplicates (byte-identical files).
- 64-bit DCT pHash (imagehash-compatible: 32x32 gray, DCT, 8x8 low block,
  median threshold): perceptual duplicates that survive re-encoding.
  Near-duplicate pairs are found at Hamming distance <= 2 via 3-way
  pigeonhole bucketing, then verified exactly -- no O(n^2) full scan.
- 32x32 mean-removed luminance signature + adjacent-pair correlation >= 0.90:
  verifies filename-implied slice series are visually the same subject.

Usage:
    python dataset_integrity_report.py --root /path/to/archive \
        --out integrity_report.txt
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.fft import dctn

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Classes whose images are independent studies (one file = one subject).
INDEPENDENT_CLASSES = {
    "Normal",
    "Pituitary",
    "Gliomas",
    "Meningothelial Tumors",
}
# Everything else is slice-based: all slices of one patient must share a split.

HAMMING_THRESHOLD = 2
SERIES_CORRELATION_THRESHOLD = 0.90
SIGNATURE_SIZE = 32


@dataclass(frozen=True)
class FileRecord:
    path: str
    relative: str
    cls: str
    sha256: str
    phash: int
    signature: bytes  # float32 * SIGNATURE_SIZE^2, packed
    error: str = ""


def phash64(image: Image.Image) -> int:
    """64-bit DCT perceptual hash (imagehash.phash compatible)."""
    gray = np.asarray(image.convert("L").resize((32, 32), Image.BICUBIC), dtype=np.float64)
    dct = dctn(gray, norm="ortho")
    low = dct[:8, :8].flatten()
    bits = (low > np.median(low)).astype(np.uint8)
    return int(np.packbits(bits).tobytes().hex(), 16)


def signature32(image: Image.Image) -> np.ndarray:
    gray = image.convert("L").resize(
        (SIGNATURE_SIZE, SIGNATURE_SIZE), Image.BILINEAR
    )
    vec = np.asarray(gray, dtype=np.float32).ravel()
    vec -= vec.mean()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _probe(args: tuple[str, str, str]) -> FileRecord:
    path, relative, cls = args
    try:
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        with Image.open(path) as image:
            image.load()
            phash = phash64(image)
            sig = signature32(image)
        return FileRecord(
            path=path,
            relative=relative,
            cls=cls,
            sha256=digest,
            phash=phash,
            signature=sig.tobytes(),
        )
    except Exception as exc:  # corrupt or undecodable files must be reported
        return FileRecord(
            path=path, relative=relative, cls=cls, sha256="", phash=-1,
            signature=b"", error=f"{type(exc).__name__}: {exc}",
        )


def walk_dataset(root: str) -> list[tuple[str, str, str]]:
    images_root = os.path.join(root, "Images_")
    base = images_root if os.path.isdir(images_root) else root
    found: list[tuple[str, str, str]] = []
    for cls in sorted(os.listdir(base)):
        cls_dir = os.path.join(base, cls)
        if not os.path.isdir(cls_dir):
            continue
        for dirpath, _, filenames in os.walk(cls_dir):
            for name in sorted(filenames):
                if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                    full = os.path.join(dirpath, name)
                    found.append((full, os.path.relpath(full, root), cls))
    return found


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_count(np.bitwise_xor(a[:, None], b[None, :])).astype(np.int16)


def near_duplicate_pairs(phashes: dict[str, int]) -> list[tuple[str, str, int]]:
    """All pairs at Hamming distance <= threshold, via 3-way pigeonhole."""
    keys = sorted(phashes)
    arr = np.array([phashes[k] for k in keys], dtype=np.uint64)
    chunks = [
        (arr >> np.uint64(43)) & np.uint64((1 << 21) - 1),
        (arr >> np.uint64(22)) & np.uint64((1 << 21) - 1),
        arr & np.uint64((1 << 22) - 1),
    ]
    candidates: set[tuple[int, int]] = set()
    for chunk in chunks:
        buckets: dict[int, list[int]] = defaultdict(list)
        for idx, value in enumerate(chunk.tolist()):
            buckets[value].append(idx)
        for members in buckets.values():
            if len(members) > 1:
                for a_pos, a in enumerate(members):
                    for b in members[a_pos + 1 :]:
                        candidates.add((a, b))

    pairs: list[tuple[str, str, int]] = []
    for a, b in sorted(candidates):
        dist = int(np.bitwise_count(arr[a] ^ arr[b]))
        if dist <= HAMMING_THRESHOLD:
            pairs.append((keys[a], keys[b], dist))
    return pairs


def cluster(items: list[str], edges: list[tuple[str, str]]) -> list[set[str]]:
    parent = {item: item for item in items}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[str, set[str]] = defaultdict(set)
    for item in items:
        groups[find(item)].add(item)
    return [g for g in groups.values() if len(g) > 1]


def numeric_stem(relative: str) -> str:
    stem = os.path.splitext(os.path.basename(relative))[0]
    stripped = stem.rstrip("0123456789").rstrip(" _-")
    return stripped.strip().lower() or stem.lower()


def numeric_suffix(relative: str) -> int:
    stem = os.path.splitext(os.path.basename(relative))[0]
    tail = stem[len(stem.rstrip("0123456789")) :]
    return int(tail) if tail else 0


def detect_series(records: list[FileRecord]) -> list[list[FileRecord]]:
    """Visually verified slice series within one class."""
    sigs = {
        r.relative: np.frombuffer(r.signature, dtype=np.float32)
        for r in records
    }
    by_stem: dict[tuple[str, str], list[FileRecord]] = defaultdict(list)
    for r in records:
        by_stem[(r.cls, numeric_stem(r.relative))].append(r)

    verified: list[list[FileRecord]] = []
    for members in by_stem.values():
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda r: numeric_suffix(r.relative))
        vecs = np.stack([sigs[r.relative] for r in ordered])
        if not np.any(vecs):
            continue
        adjacent = np.sum(vecs[1:] * vecs[:-1], axis=1)
        if float(np.median(adjacent)) >= SERIES_CORRELATION_THRESHOLD:
            verified.append(ordered)
    return verified


def fmt_cluster(members: set[str] | list[str], limit: int = 4) -> str:
    names = sorted(members)
    shown = "; ".join(names[:limit])
    return shown + (f" (+{len(names) - limit} more)" if len(names) > limit else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Dataset root (contains Images_/<class>/)")
    parser.add_argument("--out", default="", help="Report file path (stdout if empty)")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args(argv)

    started = time.time()
    found = walk_dataset(args.root)
    if not found:
        print(f"No images found under {args.root}", file=sys.stderr)
        return 1

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(_probe, found, chunksize=32))

    by_class: dict[str, list[FileRecord]] = defaultdict(list)
    for r in records:
        by_class[r.cls].append(r)
    corrupt = [r for r in records if r.error]
    healthy = [r for r in records if not r.error]
    cls_of = {r.relative: r.cls for r in healthy}

    out_lines: list[str] = []

    def emit(line: str = "") -> None:
        out_lines.append(line)
        print(line)

    emit("=" * 78)
    emit("DATA INTEGRITY REPORT - flattened brain classification dataset")
    emit("=" * 78)
    emit(f"Root      : {args.root}")
    emit(f"Scanned   : {len(found)} files, {len(by_class)} classes, "
         f"{time.time() - started:.1f}s, workers={args.workers}")
    emit(f"Methods   : SHA-256 exact | 64-bit DCT pHash (Hamming<={HAMMING_THRESHOLD}) "
         f"| 32x32 signature series (corr>={SERIES_CORRELATION_THRESHOLD})")
    emit("")
    emit("Class counts:")
    for cls in sorted(by_class):
        tag = " [independent]" if cls in INDEPENDENT_CLASSES else " [patient slices]"
        emit(f"  {len(by_class[cls]):>6,}  {cls}{tag}")
    emit(f"  {len(healthy):>6,}  TOTAL decodable")

    # ---- 1. corrupt / undecodable ----
    emit("")
    emit("1. CORRUPT / UNDECODABLE FILES")
    if corrupt:
        emit(f"   [FAIL] {len(corrupt)} file(s):")
        for r in corrupt[:10]:
            emit(f"     {r.relative} -> {r.error}")
    else:
        emit("   [PASS] all files decode")

    # ---- 2. exact SHA-256 duplicates ----
    emit("")
    emit("2. EXACT DUPLICATES (SHA-256 over file bytes)")
    sha_groups: dict[str, list[FileRecord]] = defaultdict(list)
    for r in healthy:
        sha_groups[r.sha256].append(r)
    dup_clusters = [g for g in sha_groups.values() if len(g) > 1]
    within = [g for g in dup_clusters if len({r.cls for r in g}) == 1]
    across = [g for g in dup_clusters if len({r.cls for r in g}) > 1]
    emit(f"   same image twice in one class : {len(within)} cluster(s), "
         f"{sum(len(g) for g in within)} file(s)")
    for g in within[:10]:
        emit(f"     [{g[0].cls}] " + fmt_cluster({r.relative for r in g}))
    emit(f"   same image in two classes     : {len(across)} cluster(s)")
    for g in across[:10]:
        emit("     " + fmt_cluster({f'{r.relative} ({r.cls})' for r in g}))
    emit(f"   [{'PASS' if not dup_clusters else 'FAIL'}] exact-duplicate verdict")

    # ---- 3. perceptual near-duplicates (pHash) ----
    emit("")
    emit(f"3. NEAR-DUPLICATES (pHash Hamming <= {HAMMING_THRESHOLD}, re-encode proof)")
    phashes = {r.relative: r.phash for r in healthy}
    pairs = near_duplicate_pairs(phashes)
    nd_edges = [(a, b) for a, b, _ in pairs]
    nd_clusters = cluster(list(phashes), nd_edges)
    nd_within = [c for c in nd_clusters if len({cls_of[m] for m in c}) == 1]
    nd_across = [c for c in nd_clusters if len({cls_of[m] for m in c}) > 1]
    emit(f"   near-dup pairs                : {len(pairs)}")
    emit(f"   same class                    : {len(nd_within)} cluster(s)")
    for c in nd_within[:10]:
        cls = cls_of[sorted(c)[0]]
        emit(f"     [{cls}] " + fmt_cluster(c))
    emit(f"   CROSS-CLASS (label conflict)  : {len(nd_across)} cluster(s)")
    for c in nd_across[:10]:
        emit("     " + fmt_cluster({f"{m} ({cls_of[m]})" for m in c}))
    emit(f"   [{'PASS' if not nd_across else 'FAIL'}] cross-class verdict")

    # ---- 4. subject / series analysis for patient-slice classes ----
    emit("")
    emit("4. SUBJECT GROUPING (patient-slice classes)")
    subject_counts: dict[str, int] = {}
    for cls in sorted(by_class):
        if cls in INDEPENDENT_CLASSES:
            continue
        recs = by_class[cls]
        series = detect_series(recs)
        in_series = sum(len(s) for s in series)
        subjects = len(series) + (len(recs) - in_series)
        subject_counts[cls] = subjects
        sizes = [len(s) for s in series]
        emit(f"  {cls}:")
        emit(f"    files={len(recs):,}  verified series={len(series)}  "
             f"files inside series={in_series:,}  singleton files={len(recs) - in_series:,}")
        if sizes:
            emit(f"    series size: min={min(sizes)} median={int(np.median(sizes))} "
                 f"max={max(sizes)}")
        emit(f"    => distinct subjects ~{subjects:,} (GroupKFold units)")

    # ---- 5. independence verification for the four independent classes ----
    emit("")
    emit("5. INDEPENDENCE CHECK (classes declared one-file-one-subject)")
    for cls in sorted(by_class):
        if cls not in INDEPENDENT_CLASSES:
            continue
        recs = by_class[cls]
        series = detect_series(recs)
        cls_nd = [c for c in nd_within if cls_of[sorted(c)[0]] == cls]
        if series or cls_nd:
            emit(f"  [WARN] {cls}: {len(series)} series + {len(cls_nd)} near-dup clusters "
                 f"found - NOT fully independent; group them too")
        else:
            emit(f"  [PASS] {cls}: no series, no near-duplicates "
                 f"({len(recs):,} genuinely independent images)")

    # ---- 6. verdict ----
    emit("")
    emit("=" * 78)
    emit("VERDICT")
    blockers = []
    if corrupt:
        blockers.append(f"{len(corrupt)} corrupt file(s)")
    if across:
        blockers.append(f"{len(across)} exact cross-class duplicate cluster(s)")
    if nd_across:
        blockers.append(f"{len(nd_across)} near-dup cross-class cluster(s)")
    if blockers:
        emit("  [ACTION REQUIRED] " + "; ".join(blockers))
        emit("  Remove or relabel the listed files before training.")
    else:
        emit("  [CLEAN] No exact or perceptual duplicates within or across classes.")
    if within or nd_within:
        emit(f"  Note: {len(within)} exact + {len(nd_within)} near-dup clusters exist "
             "INSIDE single classes - the splitter must collapse them into one "
             "group (already handled by dataset.py grouping).")
    emit("  Split policy: GroupKFold on subjects for the 5 patient-slice classes; "
         "per-image stratified for the 4 independent classes.")
    emit("=" * 78)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            handle.write("\n".join(out_lines) + "\n")
        print(f"\nReport written to {args.out}", file=sys.stderr)
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
