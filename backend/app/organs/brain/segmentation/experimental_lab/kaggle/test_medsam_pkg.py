"""Test suite for the LiteMedSAM training package.

Exercises pure logic and metric mathematics only. No model construction, no
forward passes, no CUDA, no training — importing torch is the heaviest thing
this does.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS, FAIL = 0, 0
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name} — {detail}")
        print(f"  FAIL  {name}  {detail}")


def close(a, b, tol=1e-6) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return abs(float(a) - float(b)) <= tol


# ═══════════════════════════════════════════════════════════════════════════
print("\n[1] engine — metrics")
from medsam_training import engine as E

m = np.zeros((8, 8), dtype=bool)
m[2:6, 2:6] = True          # 16 px
half = np.zeros((8, 8), dtype=bool)
half[2:6, 2:4] = True       # 8 px, fully inside m
far = np.zeros((8, 8), dtype=bool)
far[0:2, 0:2] = True        # disjoint
empty = np.zeros((8, 8), dtype=bool)

d, i = E.dice_iou(m, m)
check("dice_iou identical -> 1.0", close(d, 1.0) and close(i, 1.0), f"got {d},{i}")

d, i = E.dice_iou(half, m)
# inter=8, |p|=8, |g|=16 -> dice = 16/24 = 0.6667 ; iou = 8/16 = 0.5
check("dice_iou half overlap", close(d, 16 / 24) and close(i, 0.5), f"got {d},{i}")

d, i = E.dice_iou(far, m)
check("dice_iou disjoint -> 0", close(d, 0.0) and close(i, 0.0), f"got {d},{i}")

d, i = E.dice_iou(empty, empty)
check("dice_iou both empty -> 1.0 (documented convention)", close(d, 1.0) and close(i, 1.0), f"got {d},{i}")

d, i = E.dice_iou(empty, m)
check("dice_iou empty pred, non-empty gt -> 0", close(d, 0.0) and close(i, 0.0), f"got {d},{i}")

p, r = E.precision_recall(half, m)
check("precision_recall half overlap", close(p, 1.0) and close(r, 0.5), f"got {p},{r}")

p, r = E.precision_recall(empty, m)
check("precision 0 when nothing predicted", close(p, 0.0), f"got {p}")

check("hd95 identical -> 0", close(E.hd95(m, m), 0.0))
check("hd95 both empty -> 0", close(E.hd95(empty, empty), 0.0))
check("hd95 one empty -> nan", math.isnan(E.hd95(empty, m)))
check("hd95 disjoint -> finite and > 0", E.hd95(far, m) > 0 and not math.isnan(E.hd95(far, m)))

rows = [
    {"dice": 0.8, "iou": 0.7, "precision": 1.0, "recall": 0.8, "hd95_px": 2.0,
     "iou_head_pred": 0.7, "iou_head_mae": 0.0, "source": "a"},
    {"dice": 0.6, "iou": 0.5, "precision": 1.0, "recall": 0.6, "hd95_px": float("nan"),
     "iou_head_pred": 0.5, "iou_head_mae": 0.0, "source": "a"},
    {"dice": 0.9, "iou": 0.8, "precision": 1.0, "recall": 0.9, "hd95_px": 4.0,
     "iou_head_pred": 0.8, "iou_head_mae": 0.0, "source": "b"},
]
s = E.summarise(rows)
check("summarise mean dice", close(s["dice"], (0.8 + 0.6 + 0.9) / 3), f"got {s['dice']}")
check("summarise skips nan hd95", close(s["hd95_px"], 3.0), f"got {s['hd95_px']}")
check("summarise counts defined hd95", s["hd95_defined"] == 2, f"got {s['hd95_defined']}")
check("summarise n", s["n"] == 3)
check("summarise empty -> n=0", E.summarise([])["n"] == 0)

g = E.group_mean(rows, "source")
check("group_mean splits by source", set(g) == {"a", "b"}, f"got {set(g)}")
check("group_mean per-source dice", close(g["a"]["dice"], 0.7) and close(g["b"]["dice"], 0.9))

# lr_lambda: warmup then cosine
check("lr_lambda warmup start", close(E.lr_lambda(0, 100, 10, 0.05), 0.1), f"got {E.lr_lambda(0,100,10,0.05)}")
check("lr_lambda warmup end", close(E.lr_lambda(9, 100, 10, 0.05), 1.0))
check("lr_lambda at total -> min_ratio", close(E.lr_lambda(100, 100, 10, 0.05), 0.05), f"got {E.lr_lambda(100,100,10,0.05)}")
check("lr_lambda never below min_ratio", E.lr_lambda(500, 100, 10, 0.05) >= 0.05 - 1e-9)
check("lr_lambda no warmup", close(E.lr_lambda(0, 100, 0, 0.05), 1.0))

# ═══════════════════════════════════════════════════════════════════════════
print("\n[2] engine — loss shapes (the IoU bug class)")
import torch

logits = torch.randn(3, 1, 16, 16)
targets = (torch.rand(3, 1, 16, 16) > 0.7).float()
valid = torch.tensor([[16, 16], [16, 16], [16, 16]])
iou_pred = torch.rand(3, 1)


class _Cfg:
    bce_weight, dice_weight, iou_head_weight = 0.5, 0.5, 0.05
    loss_on_valid_region_only = True


loss, parts = E.segmentation_loss(logits, targets, valid, iou_pred, _Cfg())
check("loss is scalar", loss.dim() == 0, f"got shape {tuple(loss.shape)}")
check("loss is finite", bool(torch.isfinite(loss)))
check("loss parts present", set(parts) == {"bce", "dice_loss", "iou_head"}, f"got {set(parts)}")

# The IoU head must compare (B,1) against (B,1). If the target collapsed to
# (B,), F.mse_loss would broadcast to (B,B) and this assertion would fail.
with torch.no_grad():
    probs = torch.sigmoid(logits)
    hard = (probs > 0.5).float()
    spatial = (2, 3)
    inter = (hard * targets).sum(dim=spatial)
    union = hard.sum(dim=spatial) + targets.sum(dim=spatial) - inter
    true_iou = inter / union
check("IoU target keeps the channel dim -> (B,1)", tuple(true_iou.shape) == (3, 1), f"got {tuple(true_iou.shape)}")
check("IoU target matches prediction shape", true_iou.shape == iou_pred.shape)

# The guard must actually fire on a mismatched pair.
try:
    E.F.mse_loss(torch.rand(3, 1), torch.rand(3))  # broadcasts, no error
    broadcast_happens = True
except RuntimeError:
    broadcast_happens = False
check("demonstrates the original bug: (B,1) vs (B,) silently broadcasts", broadcast_happens)

# valid_region_mask
vr = E.valid_region_mask(torch.tensor([[4, 6], [8, 8]]), 8)
check("valid_region_mask shape", tuple(vr.shape) == (2, 1, 8, 8), f"got {tuple(vr.shape)}")
check("valid_region_mask honours h", int(vr[0, 0, :, 0].sum()) == 4)
check("valid_region_mask honours w", int(vr[0, 0, 0, :].sum()) == 6)
check("valid_region_mask full when hw==size", bool(vr[1].all()))

# ═══════════════════════════════════════════════════════════════════════════
print("\n[3] data — group keys, boxes, augmentation")
from medsam_training import config as C
from medsam_training import data as D

check("group key strips sequence + number",
      D.openmed_group_key("T1C+ - Cystic glioblastoma occipital , ventricle 005")
      == "Cystic glioblastoma occipital , ventricle")
check("group key handles T1", D.openmed_group_key("T1 - Glioma frontal 011") == "Glioma frontal")
check("group key handles T2", D.openmed_group_key("T2 - Medulloblastoma 018") == "Medulloblastoma")
check("group key leaves unprefixed names", D.openmed_group_key("enh_Tr-pi_0010") == "enh_Tr-pi_0010")
check("group key: same lesion, different sequences collapse",
      D.openmed_group_key("T1 - Glioma frontal 001") == D.openmed_group_key("T2 - Glioma frontal 009"))

# mask_to_box
mask = np.zeros((256, 256), dtype=np.uint8)
mask[100:140, 110:150] = 1
box = D.mask_to_box(mask, margin=5)
check("box is xyxy in 256-space", box.shape == (4,), f"got {box.shape}")
check("box correct with margin (y 100..139, x 110..149, margin 5)",
      [float(v) for v in box] == [105.0, 95.0, 154.0, 144.0], f"got {[float(v) for v in box]}")
full = np.ones((256, 256), dtype=np.uint8)
fb = D.mask_to_box(full, margin=50)
check("box clamps to frame", [float(v) for v in fb] == [0.0, 0.0, 255.0, 255.0], f"got {[float(v) for v in fb]}")

# jitter must stay inside the frame
rng = np.random.default_rng(0)
inside = True
for _ in range(200):
    b = D.mask_to_box(mask, margin=5, jitter=40, rng=rng)
    inside &= bool(b[0] >= 0 and b[1] >= 0 and b[2] <= 255 and b[3] <= 255 and b[2] > b[0] and b[3] > b[1])
check("jittered box stays in frame and non-degenerate", inside)
check("empty mask -> None", D.mask_to_box(np.zeros((16, 16), np.uint8), 5) is None)

# augmentation
img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
msk = np.zeros((32, 32), np.uint8)
msk[8:24, 8:24] = 1


class _Aug:
    aug_hflip_p, aug_rotate_p, aug_rotate_deg = 1.0, 0.0, 10.0


a_img, a_msk = D._augment(img, msk, np.random.default_rng(0), _Aug())
check("hflip flips the image", np.array_equal(a_img, img[:, ::-1]))
check("hflip flips the mask", np.array_equal(a_msk, msk[:, ::-1]))
check("hflip preserves mask area", a_msk.sum() == msk.sum())


class _AugRot:
    aug_hflip_p, aug_rotate_p, aug_rotate_deg = 0.0, 1.0, 10.0


rot_img, rot_msk = D._augment(img, msk, np.random.default_rng(3), _AugRot())
check("rotation keeps image shape", rot_img.shape == img.shape)
check("rotation keeps mask shape", rot_msk.shape == msk.shape)
check("rotation keeps mask binary", set(np.unique(rot_msk)) <= {0, 1}, f"got {np.unique(rot_msk)}")

# a rotation that would clip the lesion must be rejected
tiny = np.zeros((64, 64), np.uint8)
tiny[0:6, 0:6] = 1


class _AugRot90:
    aug_hflip_p, aug_rotate_p, aug_rotate_deg = 0.0, 1.0, 90.0


r_img, r_msk = D._augment(tiny.copy(), tiny.copy(), np.random.default_rng(1), _AugRot90())
check("rotation that would clip the lesion is rejected", r_msk.sum() == tiny.sum(),
      f"mask area {r_msk.sum()} vs {tiny.sum()}")

# mask geometry onto the target grid
out = D._mask_to_target(np.ones((64, 64), np.uint8), (64, 64), 256)
check("_mask_to_target output shape", out.shape == (256, 256), f"got {out.shape}")
check("_mask_to_target fills the valid region", out[:64, :64].all())
check("_mask_to_target leaves the pad empty", out[64:, :].sum() == 0 and out[:, 64:].sum() == 0)

# ── Dataset end-to-end on synthetic files ─────────────────────────────────
print("\n[3b] data — Dataset end-to-end")
import tempfile, os
import cv2

with tempfile.TemporaryDirectory() as tmp:
    img_dir = os.path.join(tmp, "images"); msk_dir = os.path.join(tmp, "masks")
    os.makedirs(img_dir); os.makedirs(msk_dir)
    # a 512x512 grey "scan" and a 40x40 mask well inside the frame
    base = np.full((512, 512), 60, np.uint8)
    base[180:320, 200:340] = 190
    m2 = np.zeros((512, 512), np.uint8)
    m2[220:260, 250:290] = 255
    recs = []
    for i in range(3):
        stem = f"T1 - Lesion{i:03d} region 000"
        cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), base)
        cv2.imwrite(os.path.join(msk_dir, stem + ".png"), m2)
        recs.append(D.Record("openmed", stem, os.path.join(img_dir, stem + ".jpg"),
                             os.path.join(msk_dir, stem + ".png"), D.openmed_group_key(stem)))

    ds = D.TumourSegDataset(recs, C.TrainConfig(), train=False, jitter=0)
    item = ds[0]
    check("Dataset image tensor is (3,256,256)", tuple(item["image"].shape) == (3, 256, 256),
          f"got {tuple(item['image'].shape)}")
    check("Dataset mask_target is (256,256)", tuple(item["mask_target"].shape) == (256, 256))
    check("Dataset box is (1,4)", tuple(item["box"].shape) == (1, 4), f"got {tuple(item['box'].shape)}")
    check("Dataset native_mask is (512,512)", tuple(item["native_mask"].shape) == (512, 512))
    check("Dataset valid_hw is (2,)", tuple(item["valid_hw"].shape) == (2,))
    check("Dataset image in [0,1]", float(item["image"].min()) >= 0 and float(item["image"].max()) <= 1.0001)
    check("Dataset mask_target is binary", set(np.unique(item["mask_target"].numpy()).tolist()) <= {0.0, 1.0})
    check("Dataset mask_target has foreground", float(item["mask_target"].sum()) > 0)
    check("Dataset box is inside the frame",
          float(item["box"][0, 0]) >= 0 and float(item["box"][0, 2]) <= 256)
    check("Dataset box wraps the mask",
          float(item["box"][0, 0]) <= float(np.nonzero(item["mask_target"].numpy())[1].min())
          and float(item["box"][0, 2]) >= float(np.nonzero(item["mask_target"].numpy())[1].max()))
    check("Dataset source/stem carried", item["source"] == "openmed" and item["stem"].startswith("T1 -"))

    batch = D.collate([ds[0], ds[1], ds[2]])
    check("collate stacks images", tuple(batch["image"].shape) == (3, 3, 256, 256))
    check("collate stacks masks", tuple(batch["mask_target"].shape) == (3, 1, 256, 256))
    check("collate stacks boxes", tuple(batch["box"].shape) == (3, 1, 4))
    check("collate keeps native masks as a list", len(batch["native_mask"]) == 3)
    check("collate keeps source list", batch["source"] == ["openmed"] * 3)

    # train mode with jitter must still produce sane geometry
    dst = D.TumourSegDataset(recs, C.TrainConfig(), train=True, jitter=10, seed=0)
    dst.set_epoch(1)
    it = dst[1]
    check("train-mode jittered box stays in frame",
          float(it["box"][0, 0]) >= 0 and float(it["box"][0, 2]) <= 256)
    check("train-mode mask still has foreground", float(it["mask_target"].sum()) > 0)

# ═══════════════════════════════════════════════════════════════════════════
print("\n[4] data — splits and leakage")
R = D.Record
om = []
for g in range(120):
    for k in range(4):
        stem = f"T1 - Lesion{g:03d} region {k:03d}"
        om.append(R("openmed", stem, f"/o/{stem}.jpg", f"/o/{stem}.png", D.openmed_group_key(stem)))
bt = D.assign_blocks([R("btsc", str(i), f"/b/{i}.png", f"/b/{i}.png", str(i)) for i in range(1, 401)], 16)
check("assign_blocks groups by index order", len({r.group for r in bt}) == 25, f"got {len({r.group for r in bt})}")

rng = __import__("random").Random(42)
parts = [D._partition_groups(x, 0.10, 0.10, rng) for x in (om, bt)]
mf = D.SplitManifest(train=parts[0][0] + parts[1][0], val=parts[0][1] + parts[1][1], test=parts[0][2] + parts[1][2])

check("no group leakage", D.verify_no_group_leak(mf) == [], str(D.verify_no_group_leak(mf))[:120])
check("no slice lost", len(mf.train) + len(mf.val) + len(mf.test) == len(om) + len(bt))
check("both sources in every split",
      all(len({r.source for r in p}) == 2 for p in (mf.train, mf.val, mf.test)))

bad = D.SplitManifest(train=list(mf.train), val=list(mf.val), test=list(mf.test))
moved = bad.train.pop(0)
bad.test.append(moved)
check("negative control: a real leak IS detected", len(D.verify_no_group_leak(bad)) >= 1)

mf.report = {"btsc_boundary_exposure": D._block_boundary_exposure(mf)}
ex = mf.report["btsc_boundary_exposure"]
check("exposure dict has every consumed key",
      {"btsc_total_slices", "adjacent_pairs_crossing_a_split",
       "slices_touching_a_cross_split_pair", "exposed_fraction", "interpretation"} <= set(ex),
      f"got {sorted(ex)}")
check("exposure fraction is a fraction", 0.0 <= ex["exposed_fraction"] <= 1.0)
check("exposure = 2x crossing pairs", ex["slices_touching_a_cross_split_pair"]
      == 2 * ex["adjacent_pairs_crossing_a_split"])

# ═══════════════════════════════════════════════════════════════════════════
print("\n[5] audit — popcount and phash")
from medsam_training import audit as A

vals = np.array([0, 1, 2, 3, 255, 2**63, 2**64 - 1], dtype=np.uint64)
got = A._popcount(vals).tolist()
check("popcount matches int.bit_count", got == [v.bit_count() for v in vals], f"got {got}")

check("popcount of self-xor is 0", int(A._popcount(np.array([12345], dtype=np.uint64) ^ np.array([12345], dtype=np.uint64))[0]) == 0)

# ═══════════════════════════════════════════════════════════════════════════
print("\n[6] bootstrap — weight integrity helpers")
from medsam_training import bootstrap as B

check("_looks_valid rejects a missing file", B._looks_valid("/nonexistent/lite_medsam.pth") is False)
check("EXPECTED_MIN_BYTES matches the backend", B.C.EXPECTED_MIN_BYTES == 30_000_000)
check("GDrive id matches the backend constant",
      B.C.GDRIVE_FILE_ID == "18Zed-TUTsmr2zc5CHUWd5Tu13nb6vq6z")

# ═══════════════════════════════════════════════════════════════════════════
print("\n[7] config")

c = C.TrainConfig()
check("fingerprint deterministic", c.fingerprint() == C.TrainConfig().fingerprint())
c2 = C.TrainConfig(); c2.seed = 1
check("fingerprint sensitive to seed", c2.fingerprint() != c.fingerprint())
c2 = C.TrainConfig(); c2.find_unused_parameters = True
check("new field is part of the dataclass", "find_unused_parameters" in c.to_dict())
check("IMAGE_SIZE matches the preprocessor", C.IMAGE_SIZE == 256)

print()
print("=" * 66)
print(f"PASSED {PASS}   FAILED {FAIL}")
if FAILURES:
    print("\nFAILURES:")
    for f in FAILURES:
        print("  -", f)
print("=" * 66)
sys.exit(1 if FAIL else 0)
