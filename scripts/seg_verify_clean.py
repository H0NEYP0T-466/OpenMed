"""
Final verification of a cleaned segmentation dataset.
Audits the OUTPUT directory and prints a pass/fail report.

Usage: python scripts/seg_verify_clean.py [dataset_dir]
Default: /home/honeypot/Downloads/openmed_seg_clean_strict
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/honeypot/Downloads/openmed_seg_clean_strict"
SEQ_RE = re.compile(r"^(T1C\+|T1|T2)\s*-\s*")
TRAIL_RE = re.compile(r"\s+\d+$")
MIN_INSIDE = 0.90


def group_key(stem: str) -> str:
    return SEQ_RE.sub("", TRAIL_RE.sub("", stem))


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def check(stem: str) -> dict:
    g = cv2.imread(os.path.join(ROOT, "images", stem + ".jpg"), cv2.IMREAD_GRAYSCALE)
    m = cv2.imread(os.path.join(ROOT, "masks", stem + ".png"), cv2.IMREAD_UNCHANGED)
    if g is None or m is None:
        return {"stem": stem, "ok": False}
    binm = (m > 0).astype(np.uint8)
    fg = int(binm.sum())

    thr = max(8, int(0.06 * float(g.max())))
    head = (g > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(head, 8)
    if n > 1:
        head = (lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))).astype(np.uint8)
    head = cv2.morphologyEx(head, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    h, w = head.shape
    ff = head.copy()
    ffm = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, ffm, (0, 0), 1)
    head = head | (1 - ff)

    return {
        "stem": stem, "ok": True,
        "shape_ok": g.shape == m.shape,
        "binary": set(np.unique(m).tolist()) <= {0, 255},
        "empty": fg == 0,
        "full": fg == binm.size,
        "inside": float((binm & head).sum()) / max(fg, 1),
        "fg_frac": fg / binm.size,
    }


def main() -> None:
    rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))
    stems = sorted(r["stem"] for r in rows)
    fails: list[str] = []

    print("=" * 74)
    print(f"VERIFY  {ROOT}")
    print("=" * 74)

    # 1. pair integrity
    img = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(ROOT, "images"))}
    msk = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(ROOT, "masks"))}
    n_img = len(img)
    n_msk = len(msk)
    orphan_i = len(img - msk)
    orphan_m = len(msk - img)
    print(f"1. PAIR INTEGRITY      images={n_img} masks={n_msk} "
          f"image-without-mask={orphan_i} mask-without-image={orphan_m}")
    if n_img != n_msk or orphan_i or orphan_m:
        fails.append("pair integrity")

    # 2. hashes — duplicates
    with Pool(4) as p:
        ih = p.map(sha256, [os.path.join(ROOT, "images", s + ".jpg") for s in stems], chunksize=64)
        mh = p.map(sha256, [os.path.join(ROOT, "masks", s + ".png") for s in stems], chunksize=64)
    dup_img = len(ih) - len(set(ih))
    dup_pair = len(set(zip(ih, mh))) != len(stems)
    print(f"2. DUPLICATES          duplicate image files={dup_img}  "
          f"duplicate (image,mask) pairs={'YES' if dup_pair else 'no'}")
    if dup_img or dup_pair:
        fails.append("duplicates")

    # 3. geometry + mask encoding
    with Pool(4) as p:
        recs = [r for r in p.map(check, stems, chunksize=64) if r.get("ok")]
    bad_shape = sum(1 for r in recs if not r["shape_ok"])
    non_binary = sum(1 for r in recs if not r["binary"])
    empty = sum(1 for r in recs if r["empty"])
    full = sum(1 for r in recs if r["full"])
    print(f"3. GEOMETRY/ENCODING   shape-mismatch={bad_shape} non-binary-mask={non_binary} "
          f"empty-mask={empty} full-mask={full}")
    if bad_shape or non_binary or empty or full:
        fails.append("geometry/encoding")

    # 4. alignment
    inside = np.array([r["inside"] for r in recs])
    mis = int((inside < MIN_INSIDE).sum())
    print(f"4. ALIGNMENT           inside-anatomy min={inside.min():.3f} "
          f"p1={np.percentile(inside,1):.3f} median={np.median(inside):.3f}  "
          f"below-{MIN_INSIDE}={mis}")
    if mis:
        fails.append("alignment")

    fg = np.array([r["fg_frac"] for r in recs])
    print(f"5. FOREGROUND          min={fg.min():.4f} median={np.median(fg):.4f} max={fg.max():.4f}")

    # 6. split integrity
    print("6. SPLIT INTEGRITY")
    split_of: dict[str, str] = {}
    for s in ("train", "val", "test"):
        path = os.path.join(ROOT, "splits", f"{s}.csv")
        if not os.path.isfile(path):
            print(f"   {s}: MISSING"); fails.append("split file"); continue
        names = [r["stem"] for r in csv.DictReader(open(path))]
        for n in names:
            split_of[n] = s
        print(f"   {s:5s}: {len(names):6d} slices  {len({group_key(n) for n in names}):4d} lesions")

    missing = [s for s in stems if s not in split_of]
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    for stem, sp in split_of.items():
        groups_by_split[sp].add(group_key(stem))
    overlap = ((groups_by_split["train"] & groups_by_split["val"])
               | (groups_by_split["train"] & groups_by_split["test"])
               | (groups_by_split["val"] & groups_by_split["test"]))
    print(f"   unsplit images: {len(missing)}   LEAKED lesions (in >1 split): {len(overlap)}")
    if missing or overlap:
        fails.append("split integrity")

    print("=" * 74)
    print(f"RESULT: {'CLEAN — all checks passed' if not fails else 'FAILED: ' + ', '.join(fails)}")
    print(f"        {len(stems)} pairs verified")
    print("=" * 74)


if __name__ == "__main__":
    main()
