"""
BTSC UNet segmentation dataset audit.

Dataset : /home/honeypot/Projects/FAST_API/BTSC-UNet-ViT/backend/dataset/UNet_Tumor_Dataset/
          images/*.png + masks/*.png, numeric filenames

Runs the same hygiene checks used on the OpenMed archive, adapted for this set
(greyscale images, no lesion/sequence key in the filenames), PLUS a
cross-dataset overlap check against the cleaned OpenMed set — the thing that
actually decides whether the two can be combined.

Read-only.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import cv2
import numpy as np

BTSC = "/home/honeypot/Projects/FAST_API/BTSC-UNet-ViT/backend/dataset/UNet_Tumor_Dataset"
OM = "/home/honeypot/Downloads/openmed_seg_clean_strict"
PHASH_T = 3          # calibrated on the OpenMed set
MIN_INSIDE = 0.90


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def phash(path: str, hs: int = 8, f: int = 4):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    s = hs * f
    r = cv2.resize(g, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(r)[:hs, :hs].flatten()
    b = d > np.median(d[1:])
    v = np.uint64(0)
    for x in b:
        v = (v << np.uint64(1)) | np.uint64(1 if x else 0)
    return v


def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint8)


def audit(stem: str) -> dict:
    ip = os.path.join(BTSC, "images", stem + ".png")
    mp = os.path.join(BTSC, "masks", stem + ".png")
    img = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
    msk = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
    if img is None or msk is None:
        return {"stem": stem, "ok": False}

    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binm = (msk > 0).astype(np.uint8)
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

    n_c, _, st_c, _ = cv2.connectedComponentsWithStats(binm, 8)
    areas = st_c[1:, cv2.CC_STAT_AREA] if n_c > 1 else np.array([0])
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sol = cir = None
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        a, p = cv2.contourArea(c), cv2.arcLength(c, True)
        ha = cv2.contourArea(cv2.convexHull(c))
        if a > 0 and ha > 0 and p > 0:
            sol, cir = a / ha, 4 * np.pi * a / (p * p)

    return {
        "stem": stem, "ok": True,
        "channels": 1 if img.ndim == 2 else img.shape[2],
        "shape_ok": g.shape == msk.shape,
        "binary": set(np.unique(msk).tolist()) <= {0, 255},
        "empty": fg == 0, "full": fg == binm.size,
        "fg_frac": fg / binm.size,
        "inside": float((binm & head).sum()) / max(fg, 1),
        "n_comp": int(max(n_c - 1, 0)),
        "largest_frac": float(areas.max() / fg) if fg else 0.0,
        "solidity": sol, "circularity": cir,
    }


def main() -> None:
    b_img = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(BTSC, "images"))}
    b_msk = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(BTSC, "masks"))}
    bstems = sorted(b_img & b_msk)

    print("=" * 78)
    print("BTSC UNet SEGMENTATION DATASET — AUDIT")
    print("=" * 78)
    print(f"  path: {BTSC}")
    print()
    print("1. PAIR INTEGRITY")
    print(f"   images={len(b_img)} masks={len(b_msk)} "
          f"image-without-mask={len(b_img - b_msk)} mask-without-image={len(b_msk - b_img)} "
          f"matched={len(bstems)}")
    print("   NOTE: filenames are numeric — there is NO lesion / series / sequence key "
          "in this dataset.")

    sys.stdout.flush()
    with Pool(4) as p:
        recs = [r for r in p.map(audit, bstems, chunksize=32) if r.get("ok")]

    print()
    print("2. GEOMETRY & MASK ENCODING")
    print(f"   channels        : {dict(Counter(r['channels'] for r in recs))}")
    print(f"   shape mismatch  : {sum(1 for r in recs if not r['shape_ok'])}")
    print(f"   non-binary mask : {sum(1 for r in recs if not r['binary'])}")
    print(f"   empty masks     : {sum(1 for r in recs if r['empty'])}")
    print(f"   full masks      : {sum(1 for r in recs if r['full'])}")
    fg = np.array([r["fg_frac"] for r in recs])
    print(f"   foreground      : min={fg.min():.4f} median={np.median(fg):.4f} max={fg.max():.4f}")

    ins = np.array([r["inside"] for r in recs])
    print()
    print("3. ALIGNMENT")
    print(f"   inside-anatomy  : min={ins.min():.3f} p1={np.percentile(ins,1):.3f} "
          f"median={np.median(ins):.3f}   below-{MIN_INSIDE}={int((ins < MIN_INSIDE).sum())}")

    comp = np.array([r["n_comp"] for r in recs])
    frag = sum(1 for r in recs if r["largest_frac"] < 0.90)
    sol = np.array([r["solidity"] for r in recs if r["solidity"]])
    cir = np.array([r["circularity"] for r in recs if r["circularity"]])
    print()
    print("4. MASK MORPHOLOGY")
    print(f"   connected components: median={int(np.median(comp))} max={comp.max()}")
    print(f"   fragmented (largest <90% of fg): {frag}")
    print(f"   solidity   : median={np.median(sol):.3f} p5={np.percentile(sol,5):.3f}")
    print(f"   circularity: median={np.median(cir):.3f} p95={np.percentile(cir,95):.3f} "
          f">=0.90: {(cir >= 0.90).mean()*100:.1f}%")

    # ── duplicates within BTSC ────────────────────────────────────────────
    with Pool(4) as p:
        ih = p.map(sha256, [os.path.join(BTSC, "images", s + ".png") for s in bstems], chunksize=32)
        mh = p.map(sha256, [os.path.join(BTSC, "masks", s + ".png") for s in bstems], chunksize=32)
    by = defaultdict(list)
    for s, i in zip(bstems, ih):
        by[i].append(s)
    dup_img_g = {i: v for i, v in by.items() if len(v) > 1}
    dup_img_n = sum(len(v) - 1 for v in dup_img_g.values())
    pair_g = defaultdict(list)
    for s, i, m in zip(bstems, ih, mh):
        pair_g[(i, m)].append(s)
    dup_pair_n = sum(len(v) - 1 for v in pair_g.values() if len(v) > 1)
    print()
    print("5. SHA256 DUPLICATES (within BTSC)")
    print(f"   unique image hashes : {len(set(ih))} / {len(bstems)}")
    print(f"   duplicate image files: {dup_img_n}  (in {len(dup_img_g)} groups)")
    print(f"   duplicate image+mask : {dup_pair_n}")
    print(f"   image hash == some mask hash: {len(set(ih) & set(mh))}")

    # ── perceptual duplicates within BTSC ─────────────────────────────────
    with Pool(4) as p:
        BH = np.array(p.map(phash, [os.path.join(BTSC, "images", s + ".png") for s in bstems],
                             chunksize=32), dtype=np.uint64)
    print()
    print("6. PERCEPTUAL NEAR-DUPLICATES (within BTSC)")
    print(f"   {'thr':>4} {'pairs':>8}")
    for TH in (0, 2, 3, 4, 6):
        n = 0
        CH = 512
        for st in range(0, len(BH), CH):
            d = popcount64(BH[st:st + CH, None] ^ BH[None, :])
            rs, cs = np.nonzero(d <= TH)
            n += sum(1 for r, c in zip(rs, cs) if st + r < c)
        print(f"   {TH:>4} {n:>8}")
    print("   NOTE: no series key exists, so 'same patient / adjacent slice' cannot be")
    print("         separated from 'true duplicate'. Treat these as candidates only.")

    # ── CROSS-DATASET OVERLAP ─────────────────────────────────────────────
    print()
    print("=" * 78)
    print("7. CROSS-DATASET OVERLAP WITH THE CLEANED OPENMED SET")
    print("=" * 78)
    if not os.path.isdir(OM):
        print(f"   {OM} not found — skipping.")
        return
    o_img = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(OM, "images"))}
    ostems = sorted(o_img)
    with Pool(4) as p:
        oh = p.map(sha256, [os.path.join(OM, "images", s + ".jpg") for s in ostems], chunksize=64)

    shared_sha = set(ih) & set(oh)
    print(f"   OpenMed clean set: {len(ostems)} images")
    print(f"   BTSC set         : {len(bstems)} images")
    print(f"   IDENTICAL image bytes in both (SHA256): {len(shared_sha)}")
    if shared_sha:
        bmap = {}
        for s, i in zip(bstems, ih):
            bmap.setdefault(i, []).append(s)
        omap = {}
        for s, i in zip(ostems, oh):
            omap.setdefault(i, []).append(s)
        for h in list(shared_sha)[:8]:
            print(f"     btsc {bmap[h]}  ==  openmed {omap[h][:2]}")

    # perceptual overlap
    with Pool(4) as p:
        OH = np.array(p.map(phash, [os.path.join(OM, "images", s + ".jpg") for s in ostems],
                             chunksize=64), dtype=np.uint64)
    print()
    print(f"   PERCEPTUAL overlap (pHash Hamming <= {PHASH_T}), BTSC x OpenMed:")
    hits = 0
    examples = []
    for st in range(0, len(BH), 256):
        d = popcount64(BH[st:st + 256, None] ^ OH[None, :])
        rs, cs = np.nonzero(d <= PHASH_T)
        for r, c in zip(rs, cs):
            hits += 1
            if len(examples) < 8:
                examples.append((int(d[r, c]), bstems[st + r], ostems[c]))
    print(f"   near-identical pairs: {hits}")
    for dd, b, o in examples:
        print(f"     d={dd}  btsc/{b}.png  ~  openmed/{o[:52]}.jpg")


if __name__ == "__main__":
    main()
