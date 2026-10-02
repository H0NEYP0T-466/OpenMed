"""
Strict variant of the segmentation dataset — one label per image.

Takes /home/honeypot/Downloads/openmed_seg_clean/ (already deduped on the
(image, mask) PAIR) and additionally resolves the cases where the SAME image
bytes carry DIFFERENT masks.

Rules
  A. image hash whose members all belong to one lesion  -> keep the
     best-aligned member (highest inside_frac, then largest fg_frac, then
     alphabetical). The rest are frame-duplication artefacts.
  B. image hash whose members span >1 lesion            -> drop the WHOLE
     group. The image cannot be both lesions and we cannot tell which is
     right, so it is unusable rather than merely duplicated.
  C. any surviving pair with inside_frac < 0.90         -> drop (mask not
     sitting on anatomy).

Then rebuilds the group-aware 80/10/10 split on the survivors and writes
/home/honeypot/Downloads/openmed_seg_clean_strict/.
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

SRC = "/home/honeypot/Downloads/openmed_seg_clean"
OUT = "/home/honeypot/Downloads/openmed_seg_clean_strict"
SEED = 42
SPLIT_FRACTIONS = {"train": 0.80, "val": 0.10, "test": 0.10}
MIN_INSIDE = 0.90

SEQ_RE = re.compile(r"^(T1C\+|T1|T2)\s*-\s*")
TRAIL_NUM_RE = re.compile(r"\s+\d+$")


def group_key(stem: str) -> str:
    return SEQ_RE.sub("", TRAIL_NUM_RE.sub("", stem))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rows = list(csv.DictReader(open(os.path.join(SRC, "manifest.csv"))))
    meta = {r["stem"]: r for r in rows}
    stems = [r["stem"] for r in rows]
    print(f"input pairs (already pair-deduped): {len(stems)}")
    sys.stdout.flush()

    with Pool(4) as p:
        ish = p.map(sha256, [os.path.join(SRC, "images", s + ".jpg") for s in stems], chunksize=64)

    by_img: dict[str, list[str]] = defaultdict(list)
    for s, i in zip(stems, ish):
        by_img[i].append(s)

    def rank(s: str):
        m = meta[s]
        return (-float(m["inside_frac"]), -float(m["fg_frac"]), s)

    keep: list[str] = []
    dropped_dup: list[tuple[str, str]] = []
    dropped_ambiguous: list[str] = []
    n_collapsed = n_ambiguous = 0

    for ih, members in by_img.items():
        if len(members) == 1:
            keep.append(members[0])
            continue
        if len({group_key(s) for s in members}) == 1:
            n_collapsed += 1
            best = sorted(members, key=rank)[0]
            keep.append(best)
            for s in members:
                if s != best:
                    dropped_dup.append((s, best))
        else:
            n_ambiguous += 1
            dropped_ambiguous.extend(members)

    print()
    print("RULE A — same lesion, identical image bytes")
    print(f"  groups collapsed to one pair : {n_collapsed}")
    print(f"  redundant files dropped      : {len(dropped_dup)}")
    print("RULE B — identical image bytes claiming DIFFERENT lesions")
    print(f"  ambiguous groups dropped     : {n_ambiguous}  ({len(dropped_ambiguous)} files)")

    misaligned = [s for s in keep if float(meta[s]["inside_frac"]) < MIN_INSIDE]
    keep = [s for s in keep if float(meta[s]["inside_frac"]) >= MIN_INSIDE]
    print("RULE C — mask not on anatomy")
    print(f"  misaligned dropped           : {len(misaligned)}")

    keep.sort()
    print()
    print(f"final pairs: {len(keep)}   (from {len(stems)})")

    # ── leakage-safe split ────────────────────────────────────────────────
    groups: dict[str, list[str]] = defaultdict(list)
    for s in keep:
        groups[group_key(s)].append(s)

    total = len(keep)
    targets = {k: f * total for k, f in SPLIT_FRACTIONS.items()}
    assigned: dict[str, list[str]] = {k: [] for k in SPLIT_FRACTIONS}
    counts = {k: 0 for k in SPLIT_FRACTIONS}
    rng = __import__("numpy").random.default_rng(SEED)
    for gk, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), rng.random())):
        best = max(SPLIT_FRACTIONS, key=lambda k: (targets[k] - counts[k]) / max(targets[k], 1))
        assigned[best].extend(members)
        counts[best] += len(members)

    print()
    for k in ("train", "val", "test"):
        print(f"  {k:5s}: {len(assigned[k]):6d} slices ({100*len(assigned[k])/total:5.1f}%)  "
              f"{len({group_key(s) for s in assigned[k]}):4d} lesions")
    gt = {group_key(s) for s in assigned["train"]}
    gv = {group_key(s) for s in assigned["val"]}
    gs = {group_key(s) for s in assigned["test"]}
    print(f"  LEAKAGE CHECK — lesions in >1 split: {len((gt & gv) | (gt & gs) | (gv & gs))}")

    # ── write ─────────────────────────────────────────────────────────────
    for sub in ("images", "masks", "splits"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    split_of = {s: k for k, v in assigned.items() for s in v}
    for s in keep:
        shutil.copy2(os.path.join(SRC, "images", s + ".jpg"), os.path.join(OUT, "images", s + ".jpg"))
        shutil.copy2(os.path.join(SRC, "masks", s + ".png"), os.path.join(OUT, "masks", s + ".png"))

    for k, members in assigned.items():
        with open(os.path.join(OUT, "splits", f"{k}.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["stem", "group"])
            for s in sorted(members):
                wr.writerow([s, group_key(s)])

    with open(os.path.join(OUT, "manifest.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["stem", "split", "group", "fg_frac", "inside_frac", "n_components",
                     "largest_frac", "n_tiny", "n_hole_px", "touches_border"])
        for s in keep:
            m = meta[s]
            wr.writerow([s, split_of[s], group_key(s), m["fg_frac"], m["inside_frac"],
                         m["n_components"], m["largest_frac"], m["n_tiny"],
                         m["n_hole_px"], m["touches_border"]])

    with open(os.path.join(OUT, "dropped.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["stem", "reason", "kept_as"])
        for s, kept_as in sorted(dropped_dup):
            wr.writerow([s, "duplicate image bytes (same lesion)", kept_as])
        for s in sorted(dropped_ambiguous):
            wr.writerow([s, "duplicate image bytes (different lesion — ambiguous)", ""])
        for s in sorted(misaligned):
            wr.writerow([s, f"mask off anatomy (inside_frac < {MIN_INSIDE})", ""])

    print()
    print(f"  images/ {len(keep)}   masks/ {len(keep)}   splits/ + manifest.csv + dropped.csv")
    print(f"  -> {OUT}")
    print()
    print(f"DONE. {len(stems)} -> {len(keep)} pairs. "
          f"dropped: {len(dropped_dup)} dup, {len(dropped_ambiguous)} ambiguous, {len(misaligned)} misaligned.")


if __name__ == "__main__":
    main()
