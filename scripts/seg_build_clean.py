"""
Build the cleaned, leakage-safe segmentation dataset for LiteMedSAM fine-tuning.

Input : /home/honeypot/Downloads/archive/{images,masks}
Output: /home/honeypot/Downloads/openmed_seg_clean/

Phases
  1. enumerate pairs
  2. SHA256 every image + mask (parallel)
  3. dedupe on the (image_hash, mask_hash) PAIR  — never on the image alone,
     so two identical images with different labels are both kept
  4. per-pair alignment + mask-morphology audit (parallel)
  5. group-aware 80/10/10 split, grouped on the sequence-stripped,
     slice-number-stripped stem (the lesion), balanced by slice count
  6. write images/, masks/, splits/*.csv, manifest.csv, REPORT.md

Read-only with respect to the source archive.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import cv2
import numpy as np

SRC = "/home/honeypot/Downloads/archive"
OUT = "/home/honeypot/Downloads/openmed_seg_clean"
SEED = 42
SPLIT_FRACTIONS = {"train": 0.80, "val": 0.10, "test": 0.10}

SEQ_RE = re.compile(r"^(T1C\+|T1|T2)\s*-\s*")
TRAIL_NUM_RE = re.compile(r"\s+\d+$")

WHO_KEYWORDS = [
    ("Germ Cell Tumors", ("germinoma", "germ cell")),
    ("Medulloblastoma", ("medulloblastoma",)),
    ("Meningothelial Tumors", ("meningioma",)),
    ("Schwannoma", ("schwannoma",)),
    ("Pituitary", ("pituitary",)),
    ("Mesenchymal (Non-Meningothelial Tumors)", ("solitary fibrous", "hemangiopericytoma", "sarcoma")),
    ("Mixed Neuronal and Neuronal-Glial Tumors",
     ("ganglioglioma", "neurocytoma", "dysembryoplastic", "dnet")),
    ("Gliomas", ("glioma", "glioblastoma", "astrocytoma", "oligodendroglioma", "ependymoma")),
    ("Normal", ("normal",)),
]


def series_key(stem: str) -> str:
    return TRAIL_NUM_RE.sub("", stem)


def group_key(stem: str) -> str:
    """The lesion. Same lesion under any sequence, any slice -> one group."""
    return SEQ_RE.sub("", series_key(stem))


def who_class(stem: str) -> str:
    t = group_key(stem).lower()
    for name, keys in WHO_KEYWORDS:
        if any(k in t for k in keys):
            return name
    return "(unmapped)"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_pair(stem: str) -> dict:
    """Alignment + mask morphology for one pair."""
    ip = os.path.join(SRC, "images", stem + ".jpg")
    mp = os.path.join(SRC, "masks", stem + ".png")

    g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    rec = {"stem": stem, "ok": False}

    if g is None or m is None or g.shape != m.shape:
        return rec

    binm = (m > 0).astype(np.uint8)
    fg = int(binm.sum())
    rec["ok"] = True
    rec["fg_px"] = fg
    rec["fg_frac"] = fg / binm.size

    # ---- head region: bright-ish blob, largest component, holes filled ----
    thr = max(8, int(0.06 * float(g.max())))
    head = (g > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(head, 8)
    if n > 1:
        head = (lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))).astype(np.uint8)
    head = cv2.morphologyEx(head, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    h, w = head.shape
    ff = head.copy()
    ffm = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, ffm, (0, 0), 1)
    head = head | (1 - ff)  # fill interior holes

    inside = float((binm & head).sum()) / max(fg, 1)
    rec["inside_frac"] = inside

    # ---- morphology of the mask ----
    n_c, lab_c, st_c, _ = cv2.connectedComponentsWithStats(binm, 8)
    areas = st_c[1:, cv2.CC_STAT_AREA] if n_c > 1 else np.array([0])
    rec["n_components"] = int(max(n_c - 1, 0))
    rec["largest_frac"] = float(areas.max() / fg) if fg else 0.0
    rec["n_tiny"] = int((areas < 20).sum())

    # holes inside the mask
    f2 = binm.copy()
    f2m = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(f2, f2m, (0, 0), 1)
    rec["n_hole_px"] = int((binm - (binm & f2)).sum())

    # border touching (a mask clipped by the frame edge is suspicious)
    rec["touches_border"] = bool(binm[0, :].any() or binm[-1, :].any()
                                 or binm[:, 0].any() or binm[:, -1].any())
    return rec


def main() -> None:
    img_files = sorted(f for f in os.listdir(os.path.join(SRC, "images")) if f.lower().endswith(".jpg"))
    msk_files = sorted(f for f in os.listdir(os.path.join(SRC, "masks")) if f.lower().endswith(".png"))
    img_stems = {os.path.splitext(f)[0] for f in img_files}
    msk_stems = {os.path.splitext(f)[0] for f in msk_files}
    stems = sorted(img_stems & msk_stems)

    print("=" * 78)
    print("PHASE 1-2  ENUMERATE + HASH")
    print("=" * 78)
    print(f"  pairs: {len(stems)}")
    sys.stdout.flush()
    with Pool(4) as p:
        ish = p.map(sha256, [os.path.join(SRC, "images", s + ".jpg") for s in stems], chunksize=64)
        msh = p.map(sha256, [os.path.join(SRC, "masks", s + ".png") for s in stems], chunksize=64)

    # ── PHASE 3: dedupe on the PAIR ───────────────────────────────────────
    print()
    print("=" * 78)
    print("PHASE 3  DEDUPE (on the image+mask pair, not the image alone)")
    print("=" * 78)
    pair_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for s, ih, mh in zip(stems, ish, msh):
        pair_groups[(ih, mh)].append(s)

    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    img_only_dup_conflicts = 0

    by_img: dict[str, list[str]] = defaultdict(list)
    for s, ih in zip(stems, ish):
        by_img[ih].append(s)

    for (ih, mh), members in pair_groups.items():
        members = sorted(members)
        kept.append(members[0])
        for extra in members[1:]:
            dropped.append((extra, members[0]))
        # same image bytes but a DIFFERENT mask -> not a duplicate, keep both
        same_img = by_img[ih]
        if len(same_img) > len(members):
            img_only_dup_conflicts += 1

    kept.sort()
    print(f"  identical (image, mask) pairs : {len(pair_groups)} groups")
    print(f"  unique pairs kept             : {len(kept)}")
    print(f"  redundant pairs dropped       : {len(dropped)}")
    print(f"  same image but DIFFERENT mask : {img_only_dup_conflicts}  (kept both — real label difference)")

    # ── PHASE 4: alignment + morphology ───────────────────────────────────
    print()
    print("=" * 78)
    print("PHASE 4  ALIGNMENT + MASK MORPHOLOGY  (on the kept set)")
    print("=" * 78)
    sys.stdout.flush()
    with Pool(4) as p:
        recs = p.map(audit_pair, kept, chunksize=64)
    recs = [r for r in recs if r["ok"]]
    print(f"  audited: {len(recs)}")

    inside = np.array([r["inside_frac"] for r in recs])
    misaligned = [r["stem"] for r in recs if r["inside_frac"] < 0.90]
    print(f"  mask-inside-head fraction: min={inside.min():.3f} p1={np.percentile(inside,1):.3f} "
          f"median={np.median(inside):.3f}")
    print(f"  MISALIGNED (<0.90 of the mask lies on anatomy): {len(misaligned)}")
    for s in misaligned[:5]:
        print(f"    {s}")

    comp = np.array([r["n_components"] for r in recs])
    frag = [r["stem"] for r in recs if r["largest_frac"] < 0.90]
    tiny = [r["stem"] for r in recs if r["n_tiny"] > 0]
    holed = [r["stem"] for r in recs if r["n_hole_px"] > 0]
    border = [r["stem"] for r in recs if r["touches_border"]]
    print(f"  connected components: median={int(np.median(comp))} max={comp.max()}")
    print(f"  fragmented (largest blob <90% of fg): {len(frag)}")
    print(f"  with components <20 px              : {len(tiny)}")
    print(f"  with interior holes                 : {len(holed)}")
    print(f"  mask touches the frame border       : {len(border)}")

    flags = {r["stem"]: r for r in recs}

    # ── PHASE 5: leakage-safe split ───────────────────────────────────────
    print()
    print("=" * 78)
    print("PHASE 5  GROUP-AWARE SPLIT  (grouped on the lesion, balanced by slice count)")
    print("=" * 78)
    groups: dict[str, list[str]] = defaultdict(list)
    for s in kept:
        groups[group_key(s)].append(s)

    total = len(kept)
    targets = {k: f * total for k, f in SPLIT_FRACTIONS.items()}
    assigned: dict[str, list[str]] = {k: [] for k in SPLIT_FRACTIONS}
    counts = {k: 0 for k in SPLIT_FRACTIONS}

    rng = np.random.default_rng(SEED)
    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    # shuffle ties so we are not assigning strictly largest-first within equal sizes
    order = sorted(order, key=lambda kv: (-len(kv[1]), rng.random()))

    for gk, members in order:
        best = max(SPLIT_FRACTIONS, key=lambda k: (targets[k] - counts[k]) / max(targets[k], 1))
        assigned[best].extend(members)
        counts[best] += len(members)

    print(f"  lesions (groups): {len(groups)}")
    for k in ("train", "val", "test"):
        print(f"  {k:5s}: {len(assigned[k]):6d} slices  ({100*len(assigned[k])/total:5.1f}%)  "
              f"{len({group_key(s) for s in assigned[k]}):4d} lesions")

    overlap = (set(group_key(s) for s in assigned["train"]) &
               set(group_key(s) for s in assigned["val"]) |
               set(group_key(s) for s in assigned["train"]) &
               set(group_key(s) for s in assigned["test"]) |
               set(group_key(s) for s in assigned["val"]) &
               set(group_key(s) for s in assigned["test"]))
    print(f"  LEAKAGE CHECK — lesions appearing in more than one split: {len(overlap)}")

    # ── PHASE 6: write ────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("PHASE 6  WRITE")
    print("=" * 78)
    for sub in ("images", "masks", "splits"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    split_of = {s: k for k, v in assigned.items() for s in v}
    for s in kept:
        shutil.copy2(os.path.join(SRC, "images", s + ".jpg"), os.path.join(OUT, "images", s + ".jpg"))
        shutil.copy2(os.path.join(SRC, "masks", s + ".png"), os.path.join(OUT, "masks", s + ".png"))

    for k, members in assigned.items():
        with open(os.path.join(OUT, "splits", f"{k}.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["stem", "group", "who_class"])
            for s in sorted(members):
                wr.writerow([s, group_key(s), who_class(s)])

    with open(os.path.join(OUT, "manifest.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["stem", "split", "group", "who_class", "fg_frac", "inside_frac",
                     "n_components", "largest_frac", "n_tiny", "n_hole_px", "touches_border"])
        for s in kept:
            r = flags[s]
            wr.writerow([s, split_of[s], group_key(s), who_class(s),
                         f"{r['fg_frac']:.6f}", f"{r['inside_frac']:.4f}",
                         r["n_components"], f"{r['largest_frac']:.4f}",
                         r["n_tiny"], r["n_hole_px"], int(r["touches_border"])])

    with open(os.path.join(OUT, "dropped_duplicates.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["dropped_stem", "kept_as"])
        wr.writerows(sorted(dropped))

    print(f"  images/  {len(kept)} files")
    print(f"  masks/   {len(kept)} files")
    print(f"  splits/  train.csv val.csv test.csv")
    print(f"  manifest.csv, dropped_duplicates.csv")
    print(f"  -> {OUT}")
    print()
    print("=" * 78)
    print(f"DONE. {len(stems)} source pairs -> {len(kept)} clean pairs "
          f"({len(dropped)} redundant removed). Leakage groups: {len(groups)}.")
    print("=" * 78)


if __name__ == "__main__":
    main()
