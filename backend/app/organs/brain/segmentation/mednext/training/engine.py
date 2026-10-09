"""Training engine: DDP loop, EMA, early stopping, checkpointing.

Launched once per GPU by ``torch.multiprocessing.spawn`` (``run_worker``); with
``gpus == 1`` or on CPU it runs in a single process with no DDP wrapper.

Design notes
------------
* **fp16, not bf16** — T4 is sm_75: fp16 has tensor cores, bf16 does not. Losses
  and the GRN statistic are computed in float32.
* **EMA** of the weights is evaluated next to the raw weights every epoch; the
  better of the two is what gets exported.
* **Early stopping** (patience 10) watches the macro-over-sources validation
  Dice, decided on rank 0 and broadcast so every rank leaves together.
* **Time budget** — a Kaggle session is 12 h. If another epoch would overrun
  ``time_budget_hours`` the run stops cleanly, so the final test, plots and ZIP
  still happen instead of the session dying with nothing exported.
* **Non-finite losses** are detected collectively before backward, so a NaN on
  one rank cannot desynchronise the gradient all-reduce.
* **Validation is rank-strided**, not ``DistributedSampler``, because the latter
  pads the last batch and double-counts images.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import json
import logging
import math
import os
import random
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Subset

from ..model import MedNeXtSpec, build_model, count_parameters, spec_summary, upkern_transfer
from .config import CHECKPOINT_NAME, TrainConfig
from .data import SOURCE_IDS, ArrayCache, SegDataset
from .export import build_serving_checkpoint, plain, save_serving_checkpoint
from .losses import CompoundLoss
from .metrics import aggregate, batch_stats
from .splits import SplitManifest

logger = logging.getLogger("engine")

SOURCE_NAMES = {v: k for k, v in SOURCE_IDS.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential moving average of the parameters, with a warm-up on the decay."""

    def __init__(self, model: nn.Module, decay: float, start_step: int) -> None:
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.start_step = start_step
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        if self.updates < self.start_step:
            decay = 0.0
        else:
            decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        ema_params = list(self.module.parameters())
        params = [p.detach() for p in model.parameters()]
        torch._foreach_mul_(ema_params, decay)
        torch._foreach_add_(ema_params, params, alpha=1.0 - decay)

    def state_dict(self) -> dict[str, Any]:
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.module.load_state_dict(state["module"])
        self.updates = int(state["updates"])


def learning_rate(step: int, total: int, warmup: int, base: float, floor_ratio: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base * (floor_ratio + (1.0 - floor_ratio) * cosine)


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 or "grn" in name else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_true(flag: bool, device: torch.device) -> bool:
    if not _is_dist():
        return flag
    t = torch.tensor([1.0 if flag else 0.0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item() > 0.5)


def _broadcast_flag(flag: bool, device: torch.device) -> bool:
    if not _is_dist():
        return flag
    t = torch.tensor([1.0 if flag else 0.0], device=device)
    dist.broadcast(t, src=0)
    return bool(t.item() > 0.5)


def _gather(obj: Any) -> list[Any]:
    if not _is_dist():
        return [obj]
    out: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def configure_logging(out_dir: str, rank: int) -> None:
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.FileHandler(os.path.join(out_dir, "logs", f"rank{rank}.log"))
    ]
    if rank == 0:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | %(levelname)-7s | r{rank} | %(name)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def build_network(cfg: TrainConfig) -> tuple[MedNeXtSpec, nn.Module]:
    spec = MedNeXtSpec.from_variant(
        cfg.variant, kernel_size=cfg.kernel_size, grn=cfg.grn, drop_path_rate=cfg.drop_path,
    )
    model = build_model(
        spec, deep_supervision=cfg.deep_supervision, grad_checkpoint=cfg.grad_checkpoint
    )
    if cfg.upkern_from:
        payload = torch.load(cfg.upkern_from, map_location="cpu", weights_only=True)
        source = payload["state_dict"] if "state_dict" in payload else payload
        report = upkern_transfer(model, source)
        logger.info(
            "UpKern from %s: %d copied, %d resampled, %d skipped",
            cfg.upkern_from, len(report["copied"]), len(report["resampled"]), len(report["skipped"]),
        )
    return spec, model


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: SegDataset,
    *,
    device: torch.device,
    amp: bool,
    batch_size: int,
    workers: int,
    criterion: CompoundLoss,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Rank-strided evaluation, gathered. Every rank returns the same dict."""
    rank = dist.get_rank() if _is_dist() else 0
    world = dist.get_world_size() if _is_dist() else 1
    subset = Subset(dataset, list(range(rank, len(dataset), world)))
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=min(2, workers),
        pin_memory=device.type == "cuda",
    )
    was_training = model.training
    model.eval()
    stats, sources, loss_sum, count = [], [], 0.0, 0
    for x, y, src, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        logits = logits.float()
        bce, dice = criterion.single(logits, y)
        loss_sum += float(criterion.bce_weight * bce + criterion.dice_weight * dice) * len(x)
        count += len(x)
        stats.append(batch_stats(torch.sigmoid(logits), y, threshold).cpu().numpy())
        sources.append(src.numpy())
    model.train(was_training)

    local = {
        "stats": np.concatenate(stats) if stats else np.zeros((0, 5)),
        "sources": np.concatenate(sources) if sources else np.zeros((0,), dtype=np.int64),
        "loss_sum": loss_sum, "count": count,
    }
    parts = _gather(local)
    all_stats = np.concatenate([p["stats"] for p in parts])
    all_sources = np.concatenate([p["sources"] for p in parts])
    total = sum(p["count"] for p in parts)
    summary = aggregate(all_stats, all_sources, SOURCE_NAMES)
    summary["loss"] = sum(p["loss_sum"] for p in parts) / max(1, total)
    summary["n"] = int(total)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def _make_loaders(
    cfg: TrainConfig, cache: ArrayCache, manifest: SplitManifest, rank: int, world: int
) -> tuple[SegDataset, SegDataset, DataLoader, Any]:
    train_ds = SegDataset(cache, manifest.train, train=True, cfg=cfg, seed=cfg.seed)
    val_ds = SegDataset(cache, manifest.val, train=False, cfg=cfg, seed=cfg.seed)
    if world > 1:
        sampler: Any = DistributedSampler(
            train_ds, num_replicas=world, rank=rank, shuffle=True, seed=cfg.seed, drop_last=True
        )
    else:
        sampler = RandomSampler(train_ds, generator=torch.Generator().manual_seed(cfg.seed))
    loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.workers,
        pin_memory=torch.cuda.is_available(), drop_last=True,
        persistent_workers=cfg.workers > 0, prefetch_factor=4 if cfg.workers > 0 else None,
    )
    return train_ds, val_ds, loader, sampler


def run_worker(rank: int, world_size: int, cfg_dict: dict[str, Any]) -> None:
    cfg = TrainConfig.from_dict(cfg_dict)
    out_dir = cfg.out_dir
    configure_logging(out_dir, rank)
    use_cuda = torch.cuda.is_available() and cfg.gpus > 0
    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", cfg.master_port)
        dist.init_process_group("nccl" if use_cuda else "gloo", rank=rank, world_size=world_size)
    seed_everything(cfg.seed + rank)
    is_main = rank == 0
    amp = cfg.amp and use_cuda

    try:
        _train(cfg, rank, world_size, device, amp, is_main)
    finally:
        if _is_dist():
            dist.destroy_process_group()


def _train(
    cfg: TrainConfig, rank: int, world: int, device: torch.device, amp: bool, is_main: bool
) -> None:
    out_dir = cfg.out_dir
    for sub in ("metrics", "best", "logs"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    manifest = SplitManifest.load(os.path.join(out_dir, "splits"))
    cache = ArrayCache(cfg.resolved_cache_dir())
    train_ds, val_ds, loader, sampler = _make_loaders(cfg, cache, manifest, rank, world)

    spec, core = build_network(cfg)
    core.to(device)
    ema = EMA(core, cfg.ema_decay, cfg.ema_start_step)
    model: nn.Module = DDP(core, device_ids=[rank] if device.type == "cuda" else None) if world > 1 else core
    criterion = CompoundLoss(cfg.bce_weight, cfg.dice_weight, cfg.ds_decay)
    optimizer = torch.optim.AdamW(
        parameter_groups(core, cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.99), eps=1e-8
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp, init_scale=float(cfg.amp_init_scale or 2**16)
    )

    steps_per_epoch = len(loader)
    if cfg.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, cfg.max_steps_per_epoch)
    updates_per_epoch = max(1, steps_per_epoch // cfg.grad_accum)
    total_updates = cfg.epochs * updates_per_epoch
    warmup_updates = cfg.warmup_epochs * updates_per_epoch

    if is_main:
        logger.info(spec_summary(spec, count_parameters(core)))
        logger.info(
            "device=%s world=%d amp=%s | train %d / val %d slices | %d steps/epoch, batch %d x %d GPU x accum %d",
            device, world, amp, len(train_ds), len(val_ds), steps_per_epoch,
            cfg.batch_size, world, cfg.grad_accum,
        )

    history: list[dict[str, Any]] = []
    best = {"score": -1.0, "epoch": 0, "source": "raw"}
    bad_epochs = 0
    start_epoch = 1
    global_update = 0
    elapsed_before = 0.0

    last_path = os.path.join(out_dir, "last.pt")
    if cfg.resume and os.path.isfile(cfg.resume):
        state = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        core.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        if amp and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        history, best = state["history"], state["best"]
        bad_epochs, start_epoch = state["bad_epochs"], state["epoch"] + 1
        global_update = state["global_update"]
        elapsed_before = state.get("elapsed", 0.0)
        if state.get("fingerprint") != cfg.fingerprint() and is_main:
            logger.warning("resume: config fingerprint differs from the checkpoint's")
        if is_main:
            logger.info("resumed from %s at epoch %d (best %.4f)", cfg.resume, start_epoch, best["score"])

    started = time.time()
    epoch_times: list[float] = []
    csv_path = os.path.join(out_dir, "metrics", "epochs.csv")

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_start = time.time()
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        train_ds.set_epoch(epoch)
        model.train()

        loss_sum = torch.zeros(3, device=device)
        grad_norms: list[float] = []
        skipped = 0
        optimizer.zero_grad(set_to_none=True)
        lr_now = cfg.lr

        for step, (x, y, _, _) in enumerate(loader):
            if step >= steps_per_epoch:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            boundary = (step + 1) % cfg.grad_accum == 0

            lr_now = learning_rate(
                global_update, total_updates, warmup_updates, cfg.lr, cfg.min_lr_ratio
            )
            for group in optimizer.param_groups:
                group["lr"] = lr_now

            sync = contextlib.nullcontext()
            if world > 1 and not boundary:
                sync = model.no_sync()  # type: ignore[union-attr]
            with sync:
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    outputs = model(x)
                loss, parts = criterion(outputs, y)
                if not _all_true(bool(torch.isfinite(loss)), device):
                    skipped += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss / cfg.grad_accum).backward()

            loss_sum += torch.stack([parts["total"], parts["bce"], parts["dice_loss"]])

            if boundary:
                scaler.unscale_(optimizer)
                norm = nn.utils.clip_grad_norm_(core.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(core)
                global_update += 1
                if torch.isfinite(norm):
                    grad_norms.append(float(norm))

            if is_main and (step + 1) % 50 == 0:
                logger.info(
                    "  ep %03d step %4d/%d loss %.4f lr %.2e %s",
                    epoch, step + 1, steps_per_epoch, float(loss_sum[0]) / (step + 1 - skipped),
                    lr_now,
                    f"gpu {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GB" if device.type == "cuda" else "",
                )

        done_steps = max(1, steps_per_epoch - skipped)
        if _is_dist():
            dist.all_reduce(loss_sum)
            loss_sum /= world
        train = (loss_sum / done_steps).tolist()

        raw = evaluate(core, val_ds, device=device, amp=amp, batch_size=cfg.batch_size * 2,
                       workers=cfg.workers, criterion=criterion)
        smooth = evaluate(ema.module, val_ds, device=device, amp=amp, batch_size=cfg.batch_size * 2,
                          workers=cfg.workers, criterion=criterion)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        elapsed = elapsed_before + (time.time() - started)

        stop = False
        if is_main:
            raw_score, ema_score = raw["macro_dice"], smooth["macro_dice"]
            winner, score = ("ema", ema_score) if ema_score >= raw_score else ("raw", raw_score)
            improved = score > best["score"] + cfg.min_delta
            if improved:
                best = {"score": score, "epoch": epoch, "source": winner}
                bad_epochs = 0
                _save_best(cfg, spec, core, ema, winner, epoch, raw, smooth, world, manifest)
            else:
                bad_epochs += 1

            row = {
                "epoch": epoch, "lr": lr_now,
                "train_loss": train[0], "train_bce": train[1], "train_dice_loss": train[2],
                "val_loss_raw": raw["loss"], "val_loss_ema": smooth["loss"],
                "val_dice_raw": raw["pooled_dice"], "val_dice_ema": smooth["pooled_dice"],
                "val_macro_raw": raw_score, "val_macro_ema": ema_score,
                "val_iou_ema": smooth["overall"]["iou"]["mean"],
                "val_precision_ema": smooth["overall"]["precision"]["mean"],
                "val_recall_ema": smooth["overall"]["recall"]["mean"],
                "grad_norm": float(np.mean(grad_norms)) if grad_norms else float("nan"),
                "skipped_steps": skipped, "epoch_seconds": epoch_time,
                "best_score": best["score"], "improved": int(improved),
            }
            for name in SOURCE_IDS:
                for tag, res in (("raw", raw), ("ema", smooth)):
                    entry = res["by_source"].get(name)
                    row[f"val_dice_{name}_{tag}"] = entry["dice"]["mean"] if entry else float("nan")
            history.append(row)
            _append_csv(csv_path, row)
            with open(os.path.join(out_dir, "metrics", "history.json"), "w") as handle:
                json.dump(plain(history), handle, indent=1)

            logger.info(
                "EPOCH %03d/%d | train %.4f | val raw %.4f ema %.4f (%s) | %s | lr %.2e | gn %.2f | "
                "%.1f min | best %.4f @%d (%s) | patience %d/%d%s",
                epoch, cfg.epochs, train[0], raw_score, ema_score, winner,
                " ".join(
                    f"{n}={smooth['by_source'][n]['dice']['mean']:.3f}"
                    for n in SOURCE_IDS if n in smooth["by_source"]
                ),
                lr_now, row["grad_norm"], epoch_time / 60, best["score"], best["epoch"],
                best["source"], bad_epochs, cfg.patience, "  *best*" if improved else "",
            )
            if skipped:
                logger.warning("epoch %d skipped %d non-finite steps", epoch, skipped)

            _save_last(last_path, core, ema, optimizer, scaler, epoch, history, best,
                       bad_epochs, global_update, cfg, elapsed)

            if bad_epochs >= cfg.patience:
                logger.info("EARLY STOP: no improvement > %.0e for %d epochs", cfg.min_delta, cfg.patience)
                stop = True
            budget = cfg.time_budget_hours * 3600.0
            if not stop and elapsed + 1.15 * float(np.mean(epoch_times[-3:])) > budget:
                logger.warning(
                    "TIME BUDGET: %.2f h used of %.2f h — stopping so evaluation and export still run",
                    elapsed / 3600, cfg.time_budget_hours,
                )
                stop = True

        if _broadcast_flag(stop, device):
            break

    if _is_dist():
        dist.barrier()
    if is_main:
        with open(os.path.join(out_dir, "metrics", "training_done.json"), "w") as handle:
            json.dump(plain({
                "best": best, "epochs_run": len(history), "stopped_early": bad_epochs >= cfg.patience,
                "elapsed_hours": (elapsed_before + time.time() - started) / 3600,
            }), handle, indent=1)
        logger.info("training finished: best %.4f at epoch %d (%s weights)",
                    best["score"], best["epoch"], best["source"])


def _append_csv(path: str, row: dict[str, Any]) -> None:
    new = not os.path.isfile(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def _save_best(
    cfg: TrainConfig, spec: MedNeXtSpec, core: nn.Module, ema: EMA, winner: str, epoch: int,
    raw: dict[str, Any], smooth: dict[str, Any], world: int, manifest: SplitManifest,
) -> None:
    source_module = ema.module if winner == "ema" else core
    chosen = smooth if winner == "ema" else raw
    payload = build_serving_checkpoint(
        source_module.state_dict(), spec, cfg.image_size,
        postprocess={"threshold": 0.5, "min_area_frac": 0.0, "tta_hflip": False,
                     "tuned_on_validation": False},
        metrics={"val": {k: chosen[k] for k in ("macro_dice", "pooled_dice", "overall", "by_source")}},
        training={
            "epoch": epoch, "weights": winner, "seed": cfg.seed, "variant": cfg.variant,
            "gpus": world, "split_counts": {n: len(manifest.part(n)) for n in ("train", "val", "test")},
            "config_fingerprint": cfg.fingerprint(), "stage": "best-during-training",
        },
    )
    save_serving_checkpoint(payload, os.path.join(cfg.out_dir, "best", CHECKPOINT_NAME))


def _save_last(
    path: str, core: nn.Module, ema: EMA, optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler, epoch: int, history: list[dict[str, Any]], best: dict[str, Any],
    bad_epochs: int, global_update: int, cfg: TrainConfig, elapsed: float,
) -> None:
    tmp = path + ".tmp"
    torch.save({
        "model": core.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "epoch": epoch, "history": history, "best": best, "bad_epochs": bad_epochs,
        "global_update": global_update, "fingerprint": cfg.fingerprint(), "elapsed": elapsed,
    }, tmp)
    os.replace(tmp, path)
