"""Stage 4 — metrics, loss, DDP training loop and evaluation.

Designed for **2 x Tesla T4**. Notable choices and why:

* **fp16, not bf16.** T4 is sm_75 and has no bf16 tensor cores; ``GradScaler``
  with fp16 is the fast path there. Losses are computed in float32 regardless,
  because a dice ratio in fp16 loses precision exactly where it matters.
* **Prompt encoder frozen, BatchNorm frozen.** With a box-only prompt the point
  and dense-mask branches never receive a gradient, so training them is dead
  work; and a per-rank BN running statistic would silently diverge between the
  two GPUs. Both are frozen rather than worked around with
  ``broadcast_buffers``.
* **Encoder LR is 10x lower than the decoder LR.** The TinyViT stem carries the
  pretrained prior worth preserving; the decoder is what actually needs to
  learn this dataset.
* **Validation is rank-strided, not ``DistributedSampler``.** A distributed
  sampler pads the last batch so ranks see duplicate samples, which biases the
  metric. Each rank evaluates a disjoint slice and results are gathered.

Metrics are computed at **native resolution** through the serving module's own
``postprocess_mask``, so a reported Dice is the Dice the web path would get.
HD95 is reported in **pixels** — no millimetre conversion is claimed, because
no validated pixel-spacing is available for these slices.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import sys
import time
import traceback
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from .data import SplitManifest, TumourSegDataset, collate
from .medsam_model import build_medsam_lite
from .preprocessor import MEDSAM_INPUT_SIZE, postprocess_mask

logger = logging.getLogger("engine")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (numpy, native resolution)
# ─────────────────────────────────────────────────────────────────────────────

def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Dice and IoU for two boolean masks. Both empty counts as a perfect match."""
    p = pred.astype(bool)
    g = gt.astype(bool)
    p_sum, g_sum = int(p.sum()), int(g.sum())
    if p_sum == 0 and g_sum == 0:
        return 1.0, 1.0
    inter = int((p & g).sum())
    union = p_sum + g_sum - inter
    dice = (2.0 * inter / (p_sum + g_sum)) if (p_sum + g_sum) else 1.0
    iou = (inter / union) if union else 1.0
    return float(dice), float(iou)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    p = pred.astype(bool)
    g = gt.astype(bool)
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return float(precision), float(recall)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile symmetric surface distance, in pixels.

    Returns ``nan`` when either mask is empty and the other is not — there is
    no surface to measure, and substituting a large finite number would
    masquerade as a measurement.
    """
    from scipy import ndimage

    p = pred.astype(bool)
    g = gt.astype(bool)
    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return float("nan")

    structure = ndimage.generate_binary_structure(2, 1)
    surface_p = p ^ ndimage.binary_erosion(p, structure=structure, border_value=0)
    surface_g = g ^ ndimage.binary_erosion(g, structure=structure, border_value=0)

    dt_g = ndimage.distance_transform_edt(~g)
    dt_p = ndimage.distance_transform_edt(~p)

    distances = np.concatenate([dt_g[surface_p], dt_p[surface_g]])
    if distances.size == 0:
        return float("nan")
    return float(np.percentile(distances, 95))


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-image metric rows into means, ignoring undefined values."""
    if not rows:
        return {"n": 0}

    def mean_of(key: str) -> Optional[float]:
        values: list[float] = []
        for row in rows:
            raw = row.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isnan(value):
                continue
            values.append(value)
        return float(np.mean(values)) if values else None

    out: dict[str, Any] = {
        "n": len(rows),
        "dice": mean_of("dice"),
        "iou": mean_of("iou"),
        "precision": mean_of("precision"),
        "recall": mean_of("recall"),
        "hd95_px": mean_of("hd95_px"),
        "iou_head_pred": mean_of("iou_head_pred"),
        "iou_head_mae": mean_of("iou_head_mae"),
        "hd95_defined": sum(
            1 for r in rows if r.get("hd95_px") is not None and not math.isnan(r["hd95_px"])
        ),
    }
    return out


def group_mean(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Per-source (or per-group) means, so a pooled number cannot hide a gap."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row[key]), []).append(row)
    return {name: summarise(sub) for name, sub in sorted(buckets.items())}


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def valid_region_mask(valid_hw: torch.Tensor, size: int) -> torch.Tensor:
    """(B, 1, size, size) boolean mask, True over the unpadded region."""
    batch = valid_hw.shape[0]
    device = valid_hw.device
    rows = torch.arange(size, device=device).view(1, size, 1)
    cols = torch.arange(size, device=device).view(1, 1, size)
    vh = valid_hw[:, 0].view(batch, 1, 1)
    vw = valid_hw[:, 1].view(batch, 1, 1)
    return ((rows < vh) & (cols < vw)).unsqueeze(1)


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_hw: torch.Tensor,
    iou_pred: torch.Tensor,
    cfg: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    """BCE + soft Dice, plus an IoU-head regression term.

    Everything is cast to float32 first: the autocast context produces fp16
    logits, and a dice ratio computed in fp16 loses precision precisely on the
    small foreground fractions these masks have.
    """
    logits = logits.float()
    targets = targets.float()
    eps = 1e-6

    if cfg.loss_on_valid_region_only:
        region = valid_region_mask(valid_hw, logits.shape[-1]).to(logits.dtype)
    else:
        region = torch.ones_like(targets)

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = (bce * region).sum() / region.sum().clamp_min(1.0)

    probs = torch.sigmoid(logits)
    probs = probs * region
    tgt = targets * region
    # Dice pools channel + spatial into one scalar per sample, then averages —
    # a per-sample Dice is what the metric reports, so it is what we optimise.
    all_dims = (1, 2, 3)
    intersection = (probs * tgt).sum(dim=all_dims)
    denom = probs.sum(dim=all_dims) + tgt.sum(dim=all_dims)
    dice_loss = 1.0 - (2.0 * intersection + eps) / (denom + eps)
    dice_loss = dice_loss.mean()

    total = cfg.bce_weight * bce + cfg.dice_weight * dice_loss

    parts = {"bce": float(bce.detach()), "dice_loss": float(dice_loss.detach())}

    if cfg.iou_head_weight > 0:
        with torch.no_grad():
            hard = (probs > 0.5).float()
            # Sum over the SPATIAL dims only, keeping the channel dim, so the
            # result is (B, 1) — the same shape the decoder emits. Collapsing
            # to (B,) here would make F.mse_loss broadcast (B, 1) against (B,)
            # into a (B, B) matrix, which trains the IoU head on nonsense.
            spatial = (2, 3)
            inter = (hard * tgt).sum(dim=spatial)
            union = hard.sum(dim=spatial) + tgt.sum(dim=spatial) - inter
            true_iou = ((inter + eps) / (union + eps)).detach()

        pred_iou = iou_pred.float()
        if pred_iou.shape != true_iou.shape:
            raise RuntimeError(
                f"IoU head shape mismatch: prediction {tuple(pred_iou.shape)} vs "
                f"target {tuple(true_iou.shape)}. Broadcasting here silently "
                "trains the head against a pairwise matrix."
            )
        iou_loss = F.mse_loss(pred_iou, true_iou)
        total = total + cfg.iou_head_weight * iou_loss
        parts["iou_head"] = float(iou_loss.detach())

    return total, parts


# ─────────────────────────────────────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────────────────────────────────────

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def freeze_batchnorm(model: nn.Module) -> int:
    """Keep every BatchNorm in eval mode so its statistics never move.

    Two reasons, both about correctness rather than speed:

    * Each rank would otherwise accumulate its own running mean/variance over
      its own half of the batch, and the two copies would silently diverge.
    * The pretrained statistics are part of what the checkpoint encodes. Letting
      a few hundred slices of this dataset overwrite them is a regression the
      loss will not show.

    The running statistics themselves are left intact — only the *mode* is
    pinned. Nulling them would change how the model behaves at serving time,
    where the backend calls ``.eval()`` on a single-image batch.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, _BN_TYPES):
            module.eval()
            count += 1
    return count


def build_trainable_model(cfg: Any, device: torch.device) -> nn.Module:
    """Build LiteMedSAM, load the pretrained weights, freeze what must be frozen."""
    model = build_medsam_lite()

    state = torch.load(cfg.pretrained, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    logger.info("Pretrained weights loaded with strict=True from %s", cfg.pretrained)

    if cfg.freeze_prompt_encoder:
        for param in model.prompt_encoder.parameters():
            param.requires_grad = False
        logger.info("Prompt encoder frozen (box-only prompts leave it unused).")

    if cfg.grad_checkpoint:
        for module in model.modules():
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = True
        logger.info("TinyViT gradient checkpointing enabled.")

    model.to(device)
    return model


def unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module, whether or not ``model`` is DDP-wrapped.

    ``DistributedDataParallel`` does **not** proxy attribute access to the
    module it wraps — ``ddp_model.image_encoder`` raises ``AttributeError``.
    Anywhere the architecture's own submodules are needed, go through this
    rather than assuming which form was passed in.
    """
    return model.module if isinstance(model, DDP) else model


def build_param_groups(model: nn.Module, cfg: Any) -> list[dict[str, Any]]:
    """Two groups: TinyViT encoder at ``encoder_lr``, the rest at ``lr``.

    Accepts a DDP-wrapped model or a bare one.
    """
    base = unwrap(model)
    encoder_ids = {id(p) for p in base.image_encoder.parameters()}
    encoder, rest = [], []
    for param in base.parameters():
        if not param.requires_grad:
            continue
        (encoder if id(param) in encoder_ids else rest).append(param)

    logger.info(
        "Param groups: encoder %d tensors (%.2fM) @ lr=%.1e | decoder %d tensors (%.2fM) @ lr=%.1e",
        len(encoder), sum(p.numel() for p in encoder) / 1e6, cfg.encoder_lr,
        len(rest), sum(p.numel() for p in rest) / 1e6, cfg.lr,
    )
    return [
        {"params": encoder, "lr": cfg.encoder_lr, "base_lr": cfg.encoder_lr, "name": "encoder"},
        {"params": rest, "lr": cfg.lr, "base_lr": cfg.lr, "name": "decoder"},
    ]


def lr_lambda(step: int, total: int, warmup: int, min_ratio: float) -> float:
    """Linear warmup then cosine decay to ``min_ratio`` of the base LR."""
    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: Any, steps_per_epoch: int):
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = max(0, cfg.warmup_epochs * steps_per_epoch)

    def factory(step: int) -> float:
        return lr_lambda(step, total, warmup, cfg.min_lr_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factory)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Any,
    *,
    jitter: int = 0,
    desc: str = "val",
) -> list[dict[str, Any]]:
    """Per-image metrics at native resolution, via the serving post-processor."""
    model.eval()
    rows: list[dict[str, Any]] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        boxes = batch["box"].to(device, non_blocking=True)

        with autocast_ctx(cfg, device):
            logits, iou_pred = model(images, boxes=boxes)

        logits = logits.float().cpu()
        iou_pred = iou_pred.float().cpu()

        for i in range(logits.shape[0]):
            valid_hw = batch["valid_hw"][i]
            native = batch["native_mask"][i].numpy().astype(np.uint8)
            original_hw = (int(native.shape[0]), int(native.shape[1]))
            resized_hw = (int(valid_hw[0]), int(valid_hw[1]))

            pred = postprocess_mask(
                logits[i : i + 1],
                original_hw=original_hw,
                resized_hw=resized_hw,
                target_size=MEDSAM_INPUT_SIZE,
            )
            pred_bin = pred > 0
            gt_bin = native > 0

            dice, iou = dice_iou(pred_bin, gt_bin)
            precision, recall = precision_recall(pred_bin, gt_bin)
            head = float(iou_pred[i].item())

            rows.append(
                {
                    "source": batch["source"][i],
                    "stem": batch["stem"][i],
                    "group": batch["group"][i],
                    "dice": dice,
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                    "hd95_px": hd95(pred_bin, gt_bin),
                    "iou_head_pred": head,
                    "iou_head_mae": abs(head - iou),
                    "pred_fg_frac": float(pred_bin.mean()),
                    "gt_fg_frac": float(gt_bin.mean()),
                    "jitter": jitter,
                }
            )

    model.train()
    return rows


def evaluate_distributed(
    model: nn.Module,
    dataset: TumourSegDataset,
    cfg: Any,
    device: torch.device,
    rank: int,
    world_size: int,
    *,
    jitter: int = 0,
    batch_size: int = 4,
) -> list[dict[str, Any]]:
    """Each rank evaluates a disjoint stride of the set, then results are gathered.

    Avoids ``DistributedSampler``'s padding, which would let some images be
    counted twice and skew the mean.
    """
    indices = list(range(rank, len(dataset), world_size))
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        collate_fn=collate,
    )
    local = evaluate(model, loader, device, cfg, jitter=jitter)

    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, local)
    merged: list[dict[str, Any]] = []
    for part in gathered:
        if part:
            merged.extend(part)
    merged.sort(key=lambda r: (r["source"], r["stem"], r["jitter"]))
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def autocast_ctx(cfg: Any, device: torch.device):
    """fp16 autocast on CUDA; a no-op context elsewhere.

    T4 is sm_75: fp16 has tensor cores, bf16 does not. fp16 is therefore the
    right choice here, and ``GradScaler`` handles the underflow it introduces.
    """
    if cfg.amp and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.amp.autocast("cpu", enabled=False)


def make_grad_scaler(cfg: Any, device: torch.device):
    """``torch.amp.GradScaler`` where available, ``torch.cuda.amp`` otherwise."""
    enabled = cfg.amp and device.type == "cuda"
    kwargs: dict[str, Any] = {"enabled": enabled}
    init_scale = getattr(cfg, "amp_init_scale", None)
    if init_scale:
        kwargs["init_scale"] = float(init_scale)
    try:
        return torch.amp.GradScaler("cuda", **kwargs)
    except (AttributeError, TypeError):  # torch < 2.3
        kwargs.pop("init_scale", None)
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg: Any,
    epoch: int,
    rank: int,
) -> dict[str, float]:
    model.train()
    if cfg.freeze_bn:
        freeze_batchnorm(model)

    running: dict[str, float] = {}
    steps = 0
    micro = 0                     # micro-batches since the last optimizer step
    optimizer.zero_grad(set_to_none=True)

    def _optimizer_step() -> None:
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
            )

        # GradScaler silently SKIPS optimizer.step() when the gradients
        # overflowed, and backs the scale off when it does. Advancing the
        # schedule anyway lets the LR run ahead of the parameters — which is
        # what PyTorch's "lr_scheduler.step() before optimizer.step()" warning
        # is complaining about. Detect the skip and hold the schedule.
        #
        # Detected via the scale rather than optimizer._step_count or
        # _opt_called: those are private and are not present in every version
        # (2.14 has neither, 2.10 has _opt_called).
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if scaler.get_scale() >= scale_before:
            scheduler.step()

    for step, batch in enumerate(loader):
        if cfg.max_steps_per_epoch and step >= cfg.max_steps_per_epoch:
            break

        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask_target"].to(device, non_blocking=True)
        boxes = batch["box"].to(device, non_blocking=True)
        valid_hw = batch["valid_hw"].to(device, non_blocking=True)

        with autocast_ctx(cfg, device):
            logits, iou_pred = model(images, boxes=boxes)
            loss, parts = segmentation_loss(logits, targets, valid_hw, iou_pred, cfg)
            scaled = loss / cfg.grad_accum

        # A NaN on one rank desynchronises every later collective. Detect it
        # collectively and skip the step rather than hang until the session dies.
        finite = torch.isfinite(loss).float().to(device)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() < 1.0:
            if rank == 0:
                logger.warning("Non-finite loss at epoch %d step %d — skipping step", epoch, step)
            # Drop the whole accumulation window: a poisoned micro-batch makes
            # the accumulated gradient untrustworthy, not just this term.
            optimizer.zero_grad(set_to_none=True)
            micro = 0
            continue

        scaler.scale(scaled).backward()
        micro += 1

        if micro >= cfg.grad_accum:
            _optimizer_step()
            micro = 0

        for key, value in parts.items():
            running[key] = running.get(key, 0.0) + value
        running["loss"] = running.get("loss", 0.0) + float(loss.detach())
        steps += 1

    # Flush a partial accumulation window. Without this the last
    # (len(loader) % grad_accum) micro-batches would never reach an optimizer
    # step and their gradients would be discarded.
    if micro > 0:
        _optimizer_step()

    if steps == 0:
        return {"loss": float("nan"), "steps": 0, "lr_encoder": 0.0, "lr_decoder": 0.0}

    metrics = {k: v / steps for k, v in running.items()}
    metrics["steps"] = float(steps)
    metrics["lr_encoder"] = optimizer.param_groups[0]["lr"]
    metrics["lr_decoder"] = optimizer.param_groups[-1]["lr"]
    return metrics


def reduce_scalar(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / dist.get_world_size())


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    path: str,
    epoch: int,
    best: float,
    cfg: Any,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "best": best,
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config_fingerprint": cfg.fingerprint(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
    }
    torch.save(state, path)


def save_best_weights(model: nn.Module, path: str) -> None:
    """Save a bare, CPU, unwrapped state dict — the exact shape the server loads."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {k: v.detach().cpu() for k, v in unwrap(model).state_dict().items()}
    torch.save(state, path)


# ─────────────────────────────────────────────────────────────────────────────
# DDP worker
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(out_dir: str, rank: int) -> None:
    """Rank 0 logs to stdout; every rank also gets its own file.

    Line buffering is forced on. ``mp.spawn`` children inherit a block-buffered
    stdout, so without this their output sits in a buffer and never reaches the
    notebook until the process exits — which makes a running job look hung even
    though it is training fine.
    """
    os.makedirs(out_dir, exist_ok=True)

    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):  # not a real TTY
            stream.reconfigure(line_buffering=True)

    file_handler = logging.FileHandler(os.path.join(out_dir, f"rank{rank}.log"))
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | r%(name)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    ))

    handlers: list[logging.Handler] = [file_handler]
    if rank == 0:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter(
            "%(asctime)s | r0 | %(levelname)-7s | %(name)-9s | %(message)s", datefmt="%H:%M:%S"
        ))
        handlers.append(stream_handler)

    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


def run_worker(rank: int, world_size: int, cfg_dict: dict[str, Any]) -> None:
    """Entry point for one DDP process. Called via ``torch.multiprocessing.spawn``."""
    from .config import TrainConfig

    cfg = TrainConfig(**cfg_dict)
    out_dir = cfg.out_dir
    setup_logging(out_dir, rank)

    try:
        _run_worker_body(rank, world_size, cfg)
    except Exception:
        logger.error("Rank %d failed:\n%s", rank, traceback.format_exc())
        raise


def _run_worker_body(rank: int, world_size: int, cfg: Any) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{cfg.master_port}",
        world_size=world_size,
        rank=rank,
    )
    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    torch.backends.cudnn.benchmark = True

    if rank == 0:
        logger.info("Distributed init complete: world_size=%d", world_size)

    manifest = SplitManifest.load(os.path.join(cfg.out_dir, "splits"))

    train_ds = TumourSegDataset(manifest.train, cfg, train=True, jitter=cfg.box_jitter_px, seed=cfg.seed)
    val_ds = TumourSegDataset(manifest.val, cfg, train=False, jitter=0, seed=cfg.seed)
    test_ds = TumourSegDataset(manifest.test, cfg, train=False, jitter=0, seed=cfg.seed)

    if rank == 0:
        logger.info(
            "Dataset sizes — train %d | val %d | test %d",
            len(train_ds), len(val_ds), len(test_ds),
        )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.workers > 0,
        collate_fn=collate,
    )

    model = build_trainable_model(cfg, device)
    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        # The prompt encoder is frozen, so its unused point/mask branches hold
        # no trainable parameters and DDP reports none unused either. Leaving
        # this on costs a full extra autograd traversal every iteration.
        find_unused_parameters=cfg.find_unused_parameters,
        broadcast_buffers=False,       # BN is frozen; nothing to broadcast
    )
    # After DDP wrapping and weight loading, rebuild the attention bias cache.
    model.train()
    if cfg.freeze_bn:
        freeze_batchnorm(model)

    optimizer = torch.optim.AdamW(
        build_param_groups(model, cfg),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader))
    scheduler = make_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = make_grad_scaler(cfg, device)

    start_epoch = 0
    best_score = -1.0
    epochs_without_improvement = 0

    if cfg.resume:
        if not os.path.isfile(cfg.resume):
            raise FileNotFoundError(f"--resume checkpoint not found: {cfg.resume}")
        state = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        if state.get("config_fingerprint") != cfg.fingerprint():
            raise RuntimeError(
                "Resume refused: the checkpoint was produced by a different configuration. "
                f"checkpoint={state.get('config_fingerprint')} current={cfg.fingerprint()}"
            )
        unwrap(model).load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state["best"])
        model.train()
        if cfg.freeze_bn:
            freeze_batchnorm(model)
        if rank == 0:
            logger.info("Resumed from %s at epoch %d (best=%.4f)", cfg.resume, start_epoch, best_score)

    # ── baseline: the pretrained model, before a single gradient step ────
    baseline_path = os.path.join(cfg.out_dir, "metrics", "baseline_val.json")
    if start_epoch == 0:
        if rank == 0:
            logger.info("Evaluating the pretrained checkpoint on val (baseline)...")
        baseline_rows = evaluate_distributed(model, val_ds, cfg, device, rank, world_size)
        if rank == 0:
            baseline = {
                "overall": summarise(baseline_rows),
                "per_source": group_mean(baseline_rows, "source"),
            }
            os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
            with open(baseline_path, "w") as fh:
                json.dump(baseline, fh, indent=2)
            logger.info(
                "BASELINE val macro Dice %.4f (n=%d)",
                baseline["overall"].get("dice") or float("nan"),
                baseline["overall"].get("n", 0),
            )
        dist.barrier()

    # ── training loop ────────────────────────────────────────────────────
    history: list[dict[str, Any]] = []
    epochs_path = os.path.join(cfg.out_dir, "metrics", "epochs.csv")
    best_dir = os.path.join(cfg.out_dir, "best")
    last_path = os.path.join(cfg.out_dir, "last.pt")

    if rank == 0:
        os.makedirs(os.path.dirname(epochs_path), exist_ok=True)
        with open(epochs_path, "w") as fh:
            fh.write(
                "epoch,train_loss,train_bce,train_dice_loss,train_iou_head,"
                "lr_encoder,lr_decoder,val_macro_dice,val_dice,val_iou,val_hd95_px,"
                "val_dice_openmed,val_dice_btsc,val_iou_head_mae,seconds,is_best\n"
            )

    for epoch in range(start_epoch, cfg.epochs):
        epoch_started = time.time()
        train_sampler.set_epoch(epoch)
        train_ds.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, cfg, epoch, rank
        )

        val_rows = evaluate_distributed(model, val_ds, cfg, device, rank, world_size)
        overall = summarise(val_rows)
        per_source = group_mean(val_rows, "source")
        macro_dice = float(np.mean([v["dice"] for v in per_source.values() if v.get("dice") is not None])) \
            if per_source else 0.0

        loss_mean = reduce_scalar(train_metrics.get("loss", float("nan")), device)
        macro_dice = reduce_scalar(macro_dice, device)

        improved = macro_dice > (best_score + cfg.min_delta)
        is_best = bool(improved)

        if rank == 0:
            seconds = time.time() - epoch_started
            openmed_dice = (per_source.get("openmed") or {}).get("dice")
            btsc_dice = (per_source.get("btsc") or {}).get("dice")

            logger.info(
                "epoch %3d/%d | loss %.4f | val macro-dice %.4f | val dice %.4f | "
                "hd95 %.1fpx | lr %.2e | %.0fs%s",
                epoch + 1, cfg.epochs, loss_mean, macro_dice,
                overall.get("dice") or float("nan"),
                overall.get("hd95_px") or float("nan"),
                train_metrics.get("lr_decoder", 0.0), seconds,
                "  *best*" if is_best else "",
            )

            with open(epochs_path, "a") as fh:
                fh.write(
                    f"{epoch + 1},{loss_mean:.6f},"
                    f"{train_metrics.get('bce', float('nan')):.6f},"
                    f"{train_metrics.get('dice_loss', float('nan')):.6f},"
                    f"{train_metrics.get('iou_head', float('nan')):.6f},"
                    f"{train_metrics.get('lr_encoder', float('nan')):.8f},"
                    f"{train_metrics.get('lr_decoder', float('nan')):.8f},"
                    f"{macro_dice:.6f},{overall.get('dice') or float('nan'):.6f},"
                    f"{overall.get('iou') or float('nan'):.6f},"
                    f"{overall.get('hd95_px') or float('nan'):.6f},"
                    f"{openmed_dice if openmed_dice is not None else float('nan'):.6f},"
                    f"{btsc_dice if btsc_dice is not None else float('nan'):.6f},"
                    f"{overall.get('iou_head_mae') or float('nan'):.6f},"
                    f"{seconds:.1f},{int(is_best)}\n"
                )

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": loss_mean,
                    "val_macro_dice": macro_dice,
                    "val_dice": overall.get("dice"),
                    "val_iou": overall.get("iou"),
                    "val_hd95_px": overall.get("hd95_px"),
                    "per_source": per_source,
                }
            )
            with open(os.path.join(cfg.out_dir, "metrics", "history.json"), "w") as fh:
                json.dump(history, fh, indent=2)

            if is_best:
                save_best_weights(model, os.path.join(best_dir, "lite_medsam.pth"))
                logger.info("  saved new best -> %s", os.path.join(best_dir, "lite_medsam.pth"))

        if is_best:
            best_score = macro_dice
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if rank == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, last_path,
                            epoch, best_score, cfg)

        # Early stopping decided on rank 0, broadcast so both ranks exit together.
        stop_flag = torch.tensor(
            [1 if epochs_without_improvement >= cfg.patience else 0],
            dtype=torch.int32, device=device,
        )
        dist.broadcast(stop_flag, src=0)

        if stop_flag.item() == 1:
            if rank == 0:
                logger.info(
                    "Early stopping: no improvement for %d epochs (best macro Dice %.4f).",
                    cfg.patience, best_score,
                )
            break

        dist.barrier()

    dist.barrier()

    # ── final test, once, on the best weights ────────────────────────────
    best_path = os.path.join(best_dir, "lite_medsam.pth")
    if os.path.isfile(best_path):
        unwrap(model).load_state_dict(
            torch.load(best_path, map_location="cpu", weights_only=True), strict=True
        )
        model.train()
        if cfg.freeze_bn:
            freeze_batchnorm(model)

    if rank == 0:
        logger.info("Final evaluation on the held-out test split...")

    test_rows = evaluate_distributed(model, test_ds, cfg, device, rank, world_size)

    # Prompt stress: a deliberately imperfect box, to show how much of the
    # score depends on a good prompt. This is *not* an end-to-end number —
    # the real pipeline's box comes from Grad-CAM, which this cannot simulate.
    stress_ds = TumourSegDataset(manifest.test, cfg, train=False, jitter=cfg.eval_jitter_px, seed=cfg.seed + 1)
    stress_rows = evaluate_distributed(model, stress_ds, cfg, device, rank, world_size, jitter=cfg.eval_jitter_px)

    if rank == 0:
        summary = {
            "best_val_macro_dice": best_score,
            "epochs_run": len(history),
            "test": {
                "overall": summarise(test_rows),
                "per_source": group_mean(test_rows, "source"),
            },
            "test_perturbed_prompt": {
                "jitter_px": cfg.eval_jitter_px,
                "overall": summarise(stress_rows),
                "per_source": group_mean(stress_rows, "source"),
            },
            "metric_notes": {
                "hd95_units": "pixels",
                "hd95_undefined": "nan when exactly one of pred/gt is empty",
                "prompt": (
                    "Box prompts are derived from the ground-truth mask (margin "
                    f"{cfg.box_margin_px}px, jitter +/-{cfg.eval_jitter_px}px in the "
                    "perturbed run). The deployed pipeline instead prompts with a "
                    "Grad-CAM-derived box, so these numbers are conditional "
                    "segmentation quality, not end-to-end pipeline quality."
                ),
            },
        }
        os.makedirs(os.path.join(cfg.out_dir, "metrics"), exist_ok=True)
        with open(os.path.join(cfg.out_dir, "metrics", "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        with open(os.path.join(cfg.out_dir, "metrics", "test_per_image.json"), "w") as fh:
            json.dump(test_rows, fh, indent=2)
        with open(os.path.join(cfg.out_dir, "metrics", "test_perturbed_per_image.json"), "w") as fh:
            json.dump(stress_rows, fh, indent=2)

        logger.info(
            "TEST  macro Dice %.4f | Dice %.4f | HD95 %.1fpx | perturbed Dice %.4f",
            float(np.mean([v["dice"] for v in summary["test"]["per_source"].values()])) if summary["test"]["per_source"] else float("nan"),
            summary["test"]["overall"].get("dice") or float("nan"),
            summary["test"]["overall"].get("hd95_px") or float("nan"),
            summary["test_perturbed_prompt"]["overall"].get("dice") or float("nan"),
        )

    dist.barrier()
    dist.destroy_process_group()
