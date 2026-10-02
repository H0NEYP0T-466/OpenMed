"""
Fine-tune LiteMedSAM for binary brain-tumour segmentation.

Designed to run on Kaggle, but works anywhere. Preprocessing deliberately
mirrors ``backend/app/organs/brain/segmentation/preprocessor.py`` so the
fine-tuned weights see the same input distribution the serving pipeline
produces:

  * RGB, aspect-preserving resize of the longest side to 256, INTER_AREA
  * per-image min-max normalisation to [0, 1]   <- NOT x/255
  * zero-pad the shorter side to 256x256
  * box prompt in 256x256 pixel coordinates, shape (B, 1, 4)

Training prompt: the ground-truth mask bounding box, randomly jittered. At
inference the box comes from the classifier's Grad-CAM heatmap, which is
looser and slightly off-centre, so the jitter teaches tolerance to an
imperfect box instead of overfitting to a perfect one.

Usage
-----
    python train_litemedsam.py --data <dataset_dir> --ckpt <lite_medsam.pth> \
        --out <output_dir> [--epochs 30] [--batch 16] [--smoke]

On Kaggle the paths are auto-detected under /kaggle/input and /kaggle/working.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SIZE = 256
IMAGENET_MEAN = None  # serving applies no ImageNet normalisation; do not add one


# ─────────────────────────────────────────────────────────────────────────────
# Path discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_dataset_root(explicit: str | None) -> str:
    if explicit:
        return explicit
    for base in ("/kaggle/input", "/home/honeypot/Downloads", "."):
        for cand in sorted(glob.glob(os.path.join(base, "*"))):
            if (os.path.isdir(os.path.join(cand, "images"))
                    and os.path.isdir(os.path.join(cand, "splits"))):
                return cand
    raise SystemExit("Could not locate a dataset dir containing images/ and splits/. Pass --data.")


def find_checkpoint(explicit: str | None) -> str:
    if explicit:
        return explicit
    for pat in ("/kaggle/input/**/lite_medsam.pth",
                "/home/honeypot/**/lite_medsam.pth",
                "./**/lite_medsam.pth"):
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    raise SystemExit("Could not locate lite_medsam.pth. Pass --ckpt.")


def import_model_code(dataset_root: str):
    """Import build_medsam_lite from whichever copy is reachable."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        dataset_root,
        here,
        os.path.join(os.path.dirname(here), "backend", "app", "organs", "brain", "segmentation"),
    ]
    for d in candidates:
        if os.path.isfile(os.path.join(d, "litemedsam_model.py")):
            sys.path.insert(0, d)
            from litemedsam_model import build_medsam_lite  # type: ignore
            return build_medsam_lite
        if os.path.isfile(os.path.join(d, "model.py")):
            sys.path.insert(0, d)
            from model import build_medsam_lite  # type: ignore
            return build_medsam_lite
    raise SystemExit("Could not find litemedsam_model.py / model.py next to the dataset.")


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def serve_style_image(bgr: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Exactly the serving transform. Returns (size, size, 3) float32 in [0,1]."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
    lo, hi = float(resized.min()), float(resized.max())
    resized = (resized - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(resized)
    out = np.zeros((size, size, 3), dtype=np.float32)
    out[:nh, :nw, :] = resized
    return out


def mask_bbox(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def jitter_box(box: list[float], rng: random.Random,
               expand: tuple[float, float] = (1.0, 1.30),
               shift: float = 0.10, size: int = SIZE) -> list[float]:
    """Mimic a Grad-CAM box: a little larger than the lesion, a little off-centre."""
    x0, y0, x1, y1 = box
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    e = rng.uniform(*expand)
    nw, nh = w * e, h * e
    cx += rng.uniform(-shift, shift) * nw
    cy += rng.uniform(-shift, shift) * nh
    return [max(0.0, cx - nw / 2), max(0.0, cy - nh / 2),
            min(size - 1.0, cx + nw / 2), min(size - 1.0, cy + nh / 2)]


class SegDataset(Dataset):
    def __init__(self, root: str, split: str, train: bool, limit: int | None = None):
        path = os.path.join(root, "splits", f"{split}.csv")
        with open(path) as f:
            self.stems = [r["stem"] for r in csv.DictReader(f)]
        if limit:
            self.stems = self.stems[:limit]
        self.root = root
        self.train = train
        self.rng = random.Random(1234)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, i: int):
        stem = self.stems[i]
        bgr = cv2.imread(os.path.join(self.root, "images", stem + ".jpg"), cv2.IMREAD_COLOR)
        msk = cv2.imread(os.path.join(self.root, "masks", stem + ".png"), cv2.IMREAD_GRAYSCALE)
        if bgr is None or msk is None:
            raise RuntimeError(f"unreadable pair: {stem}")

        img = serve_style_image(bgr)
        small = cv2.resize(msk, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        target = (small > 0).astype(np.float32)

        box = mask_bbox(target)
        if box is None:  # should not happen — build already dropped empty masks
            box = [0.0, 0.0, SIZE - 1.0, SIZE - 1.0]
        if self.train:
            box = jitter_box(box, self.rng)

        return (
            torch.from_numpy(img).permute(2, 0, 1),          # (3,256,256)
            torch.from_numpy(target).unsqueeze(0),            # (1,256,256)
            torch.tensor(box, dtype=torch.float32).view(1, 4),  # (1,4)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Loss & metrics
# ─────────────────────────────────────────────────────────────────────────────

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = logits.sigmoid()
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def seg_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * F.binary_cross_entropy_with_logits(logits, target) + 0.5 * dice_loss(logits, target)


@torch.no_grad()
def dice_iou(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5):
    p = (logits.sigmoid() > thr).float()
    dims = (1, 2, 3)
    inter = (p * target).sum(dim=dims)
    union = ((p + target) > 0).float().sum(dim=dims)
    dice = (2 * inter + 1e-6) / (p.sum(dim=dims) + target.sum(dim=dims) + 1e-6)
    iou = (inter + 1e-6) / (union + 1e-6)
    return dice.mean().item(), iou.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, lr_enc: float, lr_dec: float, wd: float):
    enc, dec = [], []
    for name, p in model.named_parameters():
        (enc if name.startswith("image_encoder") else dec).append(p)
    return torch.optim.AdamW(
        [{"params": enc, "lr": lr_enc}, {"params": dec, "lr": lr_dec}],
        weight_decay=wd,
    )


def cosine_with_warmup(opt, warmup: int, total: int, min_frac: float = 0.05):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        prog = (step - warmup) / max(total - warmup, 1)
        return min_frac + (1 - min_frac) * 0.5 * (1 + np.cos(np.pi * min(prog, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, preview=None):
    model.eval()
    ds, ious, n = 0.0, 0.0, 0
    for img, msk, box in loader:
        img, msk, box = img.to(device), msk.to(device), box.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, _ = model(img, boxes=box)
        d, i = dice_iou(logits.float(), msk)
        ds += d * img.size(0)
        ious += i * img.size(0)
        n += img.size(0)
        if preview is not None and len(preview["img"]) < 4:
            preview["img"].append(img[:1].float().cpu())
            preview["gt"].append(msk[:1].float().cpu())
            preview["pr"].append((logits[:1].float().sigmoid() > 0.5).float().cpu())
    model.train()
    return ds / max(n, 1), ious / max(n, 1)


def save_preview(preview, path):
    if not preview["img"]:
        return
    rows = []
    for img, gt, pr in zip(preview["img"], preview["gt"], preview["pr"]):
        rgb = (img[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        overlay = rgb.copy()
        overlay[pr[0, 0].numpy() > 0] = [237, 111, 92]   # coral, Atelier Zero accent
        row = np.concatenate([rgb, np.stack([gt[0, 0].numpy()] * 3, -1).astype(np.uint8) * 255,
                              overlay], axis=1)
        rows.append(row)
    cv2.imwrite(path, cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="/kaggle/working" if os.path.isdir("/kaggle/working") else "./run")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr-enc", type=float, default=1e-5)
    ap.add_argument("--lr-dec", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch — pipeline check")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    root = find_dataset_root(args.data)
    ckpt = find_checkpoint(args.ckpt)
    build_medsam_lite = import_model_code(root)
    os.makedirs(args.out, exist_ok=True)

    print("=" * 74)
    print("LiteMedSAM fine-tune")
    print("=" * 74)
    print(f"  dataset : {root}")
    print(f"  ckpt    : {ckpt}")
    print(f"  out     : {args.out}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device  : {device}")
    amp_dtype = torch.float16 if device.type == "cuda" else None

    limit = 64 if args.smoke else None
    epochs = 1 if args.smoke else args.epochs
    train_ds = SegDataset(root, "train", train=True, limit=limit)
    val_ds = SegDataset(root, "val", train=False, limit=limit)
    test_ds = SegDataset(root, "test", train=False, limit=limit)
    print(f"  splits  : train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=(device.type == "cuda"),
                          drop_last=True, persistent_workers=args.workers > 0)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"))
    test_ld = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                         num_workers=args.workers, pin_memory=(device.type == "cuda"))

    model = build_medsam_lite().to(device)
    state = torch.load(ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  weights : loaded (missing={len(missing)} unexpected={len(unexpected)})")
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  params  : {n_par/1e6:.2f}M")

    opt = build_optimizer(model, args.lr_enc, args.lr_dec, args.wd)
    total = max(1, len(train_ld) * epochs)
    sched = cosine_with_warmup(opt, args.warmup, total)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype is not None)

    best = -1.0
    best_path = os.path.join(args.out, "litemedsam_finetuned.pth")
    history = []
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for step, (img, msk, box) in enumerate(train_ld, 1):
            img, msk, box = img.to(device), msk.to(device), box.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits, _ = model(img, boxes=box)
                loss = seg_loss(logits.float(), msk)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            if step % 50 == 0:
                print(f"  ep{ep:02d} step {step:4d}/{len(train_ld)}  loss={run/step:.4f}  "
                      f"lr={opt.param_groups[1]['lr']:.2e}", flush=True)

        vd, vi = evaluate(model, val_ld, device, amp_dtype)
        history.append({"epoch": ep, "train_loss": run / max(len(train_ld), 1),
                        "val_dice": vd, "val_iou": vi})
        print(f"  ep{ep:02d} DONE  train_loss={run/max(len(train_ld),1):.4f}  "
              f"val_dice={vd:.4f}  val_iou={vi:.4f}  ({(time.time()-t0)/60:.1f} min)", flush=True)

        if vd > best:
            best = vd
            torch.save(model.state_dict(), best_path)
            print(f"        saved best (val_dice={vd:.4f}) -> {best_path}", flush=True)

    print()
    print("=" * 74)
    print("FINAL")
    print("=" * 74)
    model.load_state_dict(torch.load(best_path, map_location=device))
    preview = {"img": [], "gt": [], "pr": []}
    td, ti = evaluate(model, test_ld, device, amp_dtype, preview=preview)
    print(f"  best val_dice : {best:.4f}")
    print(f"  TEST  dice={td:.4f}  iou={ti:.4f}")
    save_preview(preview, os.path.join(args.out, "predictions_preview.png"))

    with open(os.path.join(args.out, "training_history.json"), "w") as f:
        json.dump({"args": vars(args), "best_val_dice": best,
                   "test_dice": td, "test_iou": ti, "history": history}, f, indent=2)
    print(f"  wrote {best_path}")
    print(f"  wrote {os.path.join(args.out, 'training_history.json')}")
    print(f"  wrote {os.path.join(args.out, 'predictions_preview.png')}")


if __name__ == "__main__":
    main()
