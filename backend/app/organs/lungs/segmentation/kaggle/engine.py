"""Training engine: DDP loop, EMA, early stopping, loud resumability.

Launched once per GPU by ``torch.multiprocessing.spawn`` (``run_worker``); with one
GPU or on CPU (wiring checks only) it runs in a single process with no DDP wrapper.

Design notes
------------
* **fp16, not bf16.** T4 is sm_75: fp16 has tensor cores, bf16 does not. Losses are
  computed in float32 and autocast keeps norm statistics in float32.
* **EMA** (0.999) is evaluated next to the raw weights every epoch; the better of the
  two is exported and the export records which it was.
* **Two passes.** Head A on the full frame for every sample; head B on the lung crop of
  supervised samples only (about 13 % of the pool). One ``forward(full, crop)`` call
  keeps a single autograd graph per step, which is what DDP needs.
* **Non-finite losses** are detected collectively before backward, so one rank cannot
  desynchronise the gradient all-reduce.
* **Resume is loud.** Exactly one of ``FRESH START``, ``RESUMED from epoch N`` or
  ``WEIGHTS-ONLY RESUME`` is logged, with the parts that were restored. A ``last``
  checkpoint (full state: model, EMA, optimiser, schedule position, scaler, RNG states
  of every rank, history) is kept apart from ``best``; granularity is the epoch.
* **Time budget.** If another epoch would overrun ``time_budget_hours`` the run stops
  cleanly so the final test, plots and ZIP still happen.
* **Validation is rank-strided**, not ``DistributedSampler``, which pads the last batch
  and double-counts images.
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
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from ..checkpoint import CHECKPOINT_NAME, build_serving_checkpoint, plain, save_serving_checkpoint
from ..labels import CLASS_COVID, CLASS_IDS, CLASS_NORMAL
from ..model import LungSegNet, SegSpec, build_model, count_parameters
from ..postprocess import LungPost, OpacityPost
from ..preprocessor import normalise_u8
from . import splits as splits_mod
from .config import TrainConfig
from .data import ArrayCache, EpochBatchSampler, SegDataset, collate
from .losses import TwoHeadLoss
from .metrics import VAL_FP_MIN_FRACTION, batch_stats, scores_from_stats

logger = logging.getLogger("engine")

COVID_ID = CLASS_IDS[CLASS_COVID]
NORMAL_ID = CLASS_IDS[CLASS_NORMAL]


# ---------------------------------------------------------------------------- helpers


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
        decay = 0.0 if self.updates < self.start_step else min(
            self.decay, (1 + self.updates) / (10 + self.updates))
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


def parameter_groups(model: nn.Module, cfg: TrainConfig) -> list[dict[str, Any]]:
    """Encoder and new layers train at different rates; norm and bias are exempt from decay."""
    buckets: dict[tuple[bool, bool], list[nn.Parameter]] = {
        (enc, dec): [] for enc in (True, False) for dec in (True, False)}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = param.ndim <= 1 or name.endswith(".bias")
        buckets[(name.startswith("encoder."), not no_decay)].append(param)
    groups = []
    for (is_encoder, decays), params in buckets.items():
        if params:
            groups.append({"params": params, "weight_decay": cfg.weight_decay if decays else 0.0,
                           "lr_mult": cfg.encoder_lr_mult if is_encoder else 1.0})
    return groups


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


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        with contextlib.suppress(Exception):  # device count may differ after a restart
            torch.cuda.set_rng_state_all(state["cuda"])


def configure_logging(out_dir: str, rank: int) -> None:
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.FileHandler(os.path.join(out_dir, "logs", f"rank{rank}.log"), encoding="utf-8")]
    if rank == 0:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, force=True, handlers=handlers, datefmt="%H:%M:%S",
                        format=f"%(asctime)s | %(levelname)-7s | r{rank} | %(name)-8s | %(message)s")


def build_spec(cfg: TrainConfig) -> SegSpec:
    return SegSpec(encoder=cfg.encoder, heads=cfg.heads(), drop_path=cfg.drop_path,
                   decoder_dropout=cfg.decoder_dropout, head_dropout=cfg.head_dropout)


def pretrained_source(cfg: TrainConfig) -> str | None:
    """ImageNet weights for the encoder, or a loud refusal. Never a silent random start."""
    if cfg.encoder != "mit-b0" or cfg.init_from:
        return None
    if cfg.pretrained:
        return cfg.pretrained
    if cfg.allow_random_init:
        return None
    raise RuntimeError(
        "No pretrained encoder weights configured. The design calls for ImageNet-pretrained "
        "MiT-B0 ('nvidia/mit-b0', needs internet on Kaggle or a dataset copy passed with "
        "--pretrained <dir>). Pass --allow-random-init only to knowingly train from scratch."
    )


def build_network(cfg: TrainConfig) -> tuple[SegSpec, LungSegNet]:
    spec = build_spec(cfg)
    return spec, build_model(spec, pretrained_source=pretrained_source(cfg))


def selection_score(res: dict[str, Any], cfg: TrainConfig, has_opacity: bool) -> tuple[float, float | None]:
    """Validation score used for early stopping and weight choice.

    Lung stage: lung Dice. Two-head: ``share * lung + (1 - share) * opacity`` where
    ``opacity = positive Dice - fpr_weight * Normal false-positive rate``, so a head that
    paints opacity everywhere cannot win on Dice alone.
    """
    lung = float(res["lung_dice"])
    if not has_opacity:
        return lung, None
    pos = res.get("opacity_pos_dice")
    fpr = res.get("normal_fpr")
    opacity = max(0.0, (pos or 0.0) - cfg.fpr_weight * (fpr or 0.0))
    return cfg.selection_lung_share * lung + (1.0 - cfg.selection_lung_share) * opacity, opacity


# ------------------------------------------------------------------------- validation


@torch.no_grad()
def evaluate(
    model: nn.Module, dataset: SegDataset, *, device: torch.device, amp: bool, batch_size: int,
    workers: int, criterion: TwoHeadLoss, has_opacity: bool,
) -> dict[str, Any]:
    """Rank-strided evaluation, gathered. Every rank returns the same dict."""
    rank = dist.get_rank() if _is_dist() else 0
    world = dist.get_world_size() if _is_dist() else 1
    subset = Subset(dataset, list(range(rank, len(dataset), world)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate,
                        num_workers=min(2, workers), pin_memory=device.type == "cuda")
    was_training = model.training
    model.eval()
    lung_stats: list[np.ndarray] = []
    op_stats: list[np.ndarray] = []
    op_cls: list[np.ndarray] = []
    loss_sum, count = 0.0, 0
    for batch in loader:
        full = batch["full"].to(device, non_blocking=True)
        lung_t = batch["lung"].to(device, non_blocking=True)
        sup = batch["sup"].to(device)
        idx = sup.nonzero().squeeze(1) if has_opacity else None
        crop = opacity_t = None
        if has_opacity and idx is not None and idx.numel():
            crop = normalise_u8(batch["crop"].to(device)[idx])
            opacity_t = batch["opacity"].to(device)[idx]
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            lung_logits, op_logits = model(normalise_u8(full), crop)
        lung_logits = lung_logits.float()
        loss, _ = criterion(lung_logits, lung_t, None if op_logits is None else op_logits.float(),
                            opacity_t)
        loss_sum += float(loss) * len(full)
        count += len(full)
        lung_stats.append(batch_stats(torch.sigmoid(lung_logits), lung_t).cpu().numpy())
        if op_logits is not None and opacity_t is not None and idx is not None:
            op_stats.append(batch_stats(torch.sigmoid(op_logits.float()), opacity_t).cpu().numpy())
            op_cls.append(batch["cls"].to(device)[idx].cpu().numpy())
    model.train(was_training)

    local = {"lung": np.concatenate(lung_stats) if lung_stats else np.zeros((0, 5)),
             "op": np.concatenate(op_stats) if op_stats else np.zeros((0, 5)),
             "cls": np.concatenate(op_cls) if op_cls else np.zeros((0,), dtype=np.int64),
             "loss": loss_sum, "count": count}
    parts = _gather(local)
    lung_all = np.concatenate([p["lung"] for p in parts])
    op_all = np.concatenate([p["op"] for p in parts])
    cls_all = np.concatenate([p["cls"] for p in parts])
    total = sum(p["count"] for p in parts)

    lung_scores = scores_from_stats(lung_all)
    result: dict[str, Any] = {
        "lung_dice": float(lung_scores["dice"].mean()), "lung_iou": float(lung_scores["iou"].mean()),
        "loss": sum(p["loss"] for p in parts) / max(1, total), "n": int(total),
        "opacity_pos_dice": None, "opacity_pos_iou": None, "normal_fpr": None,
        "normal_mean_fraction": None, "n_pos": 0, "n_normal": 0,
    }
    if len(op_all):
        scores = scores_from_stats(op_all)
        pos = cls_all == COVID_ID
        normal = cls_all == NORMAL_ID
        result["n_pos"], result["n_normal"] = int(pos.sum()), int(normal.sum())
        if pos.any():
            result["opacity_pos_dice"] = float(scores["dice"][pos].mean())
            result["opacity_pos_iou"] = float(scores["iou"][pos].mean())
        if normal.any():
            fraction = op_all[normal, 3] / float(dataset[0]["full"].numel())
            result["normal_fpr"] = float((fraction >= VAL_FP_MIN_FRACTION).mean())
            result["normal_mean_fraction"] = float(fraction.mean())
    return result


# ----------------------------------------------------------------------------- worker


def _subsample(records: list[Any], frac: float, seed: int) -> list[Any]:
    """Probe runs: a stratified fraction that keeps every (supervision, class) group."""
    rng = np.random.default_rng(seed)
    groups: dict[tuple[bool, str], list[Any]] = {}
    for r in records:
        groups.setdefault((r.opacity_supervised, r.cls), []).append(r)
    kept: list[Any] = []
    for members in groups.values():
        take = max(2, round(len(members) * frac))
        pick = rng.choice(len(members), size=min(take, len(members)), replace=False)
        kept.extend(members[int(i)] for i in sorted(pick))
    return kept


def _make_loaders(
    cfg: TrainConfig, cache: ArrayCache, split: splits_mod.SplitResult, rank: int, world: int,
    has_opacity: bool,
) -> tuple[SegDataset, SegDataset, DataLoader, EpochBatchSampler]:
    train_records = split.part("train")
    if cfg.is_probe:
        train_records = _subsample(train_records, cfg.probe_train_frac, cfg.seed)
    train_ds = SegDataset(cache, train_records, train=True, cfg=cfg, want_crops=has_opacity)
    val_ds = SegDataset(cache, split.part("val"), train=False, cfg=cfg, want_crops=has_opacity)
    sampler = EpochBatchSampler(
        train_ds.sup if has_opacity else np.zeros(len(train_ds), dtype=bool), train_ds.cls,
        batch_size=cfg.batch_size, world=world, rank=rank, seed=cfg.seed,
        min_supervised=cfg.min_supervised_per_batch if has_opacity else 0,
        normal_neg_weight=cfg.normal_neg_weight if has_opacity else 1.0,
    )
    loader = DataLoader(
        train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=cfg.workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=cfg.workers > 0,
        prefetch_factor=4 if cfg.workers > 0 else None,
    )
    return train_ds, val_ds, loader, sampler


def run_worker(rank: int, world_size: int, cfg_dict: dict[str, Any]) -> None:
    cfg = TrainConfig.from_dict(cfg_dict)
    configure_logging(cfg.out_dir, rank)
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
    try:
        _train(cfg, rank, world_size, device, cfg.amp and use_cuda, rank == 0)
    finally:
        if _is_dist():
            dist.destroy_process_group()


def _resume_path(cfg: TrainConfig, last_path: str) -> str | None:
    if cfg.resume == "none":
        return None
    if cfg.resume == "auto":
        return last_path if os.path.isfile(last_path) else None
    if not os.path.isfile(cfg.resume):
        raise FileNotFoundError(f"--resume {cfg.resume}: no such checkpoint")
    return cfg.resume


def load_weights_only(core: LungSegNet, ema: EMA, path: str) -> dict[str, list[str]]:
    """Restore from a serving checkpoint. Missing heads stay freshly initialised, loudly."""
    from ..checkpoint import load_serving_checkpoint

    payload = load_serving_checkpoint(path)
    state = payload["state_dict"]
    present = {k.split(".", 1)[0] for k in state}
    wanted = {"encoder", "decoder", "head_lung"} | ({"head_opacity"} if core.has_opacity else set())
    restored = sorted(wanted & present)
    fresh = sorted(wanted - present)
    if "head_opacity" in present and not core.has_opacity:
        raise RuntimeError(f"{path} has an opacity head but this run trains the lung stage only.")
    partial = {k: v for k, v in state.items() if k.split(".", 1)[0] in restored}
    # canonical -> installed layout, then a non-strict load restricted to the restored parts
    encoder_state = {k.split(".", 1)[1]: v for k, v in partial.items() if k.startswith("encoder.")}
    if encoder_state:
        core.encoder.load_canonical(encoder_state)
    for name in ("decoder", "head_lung", "head_opacity"):
        module = getattr(core, name, None)
        if module is None or name not in restored:
            continue
        module.load_state_dict({k.split(".", 1)[1]: v for k, v in partial.items()
                                if k.startswith(name + ".")}, strict=True)
    ema.module.load_state_dict(core.state_dict())
    return {"restored": restored, "fresh": fresh}


def _train(
    cfg: TrainConfig, rank: int, world: int, device: torch.device, amp: bool, is_main: bool
) -> None:
    out_dir = cfg.out_dir
    for sub in ("best", "logs"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    split = splits_mod.load_manifest(os.path.join(out_dir, "split_manifest.csv"))
    cache = ArrayCache(cfg.resolved_cache_dir())
    has_opacity = "opacity" in cfg.heads()
    train_ds, val_ds, loader, sampler = _make_loaders(cfg, cache, split, rank, world, has_opacity)

    spec, core = build_network(cfg)
    core.to(device)
    ema = EMA(core, cfg.ema_decay, cfg.ema_start_step)
    model: nn.Module = (DDP(core, device_ids=[rank] if device.type == "cuda" else None)
                        if world > 1 else core)
    criterion = TwoHeadLoss(lung_weight=cfg.lung_weight, opacity_weight=cfg.opacity_weight,
                            bce_weight=cfg.bce_weight, dice_weight=cfg.dice_weight,
                            dice_eps=cfg.dice_eps)
    optimizer = torch.optim.AdamW(parameter_groups(core, cfg), lr=cfg.lr, betas=(0.9, 0.999),
                                  eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=float(cfg.amp_init_scale))

    steps_per_epoch = len(sampler)
    if cfg.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, cfg.max_steps_per_epoch)
    if steps_per_epoch < 1:
        raise RuntimeError(
            f"Zero steps per epoch: {len(train_ds)} training images with batch {cfg.batch_size} x "
            f"{world} GPU(s). Lower --batch.")
    updates_per_epoch = max(1, steps_per_epoch // cfg.grad_accum)
    total_updates = cfg.epochs * updates_per_epoch
    warmup_updates = cfg.warmup_epochs * updates_per_epoch

    history: list[dict[str, Any]] = []
    best = {"score": -1.0, "epoch": 0, "source": "raw"}
    bad_epochs, start_epoch, global_update, elapsed_before = 0, 1, 0, 0.0
    last_path = os.path.join(out_dir, "last.pt")
    csv_path = os.path.join(out_dir, "epochs.csv")

    resume_from = _resume_path(cfg, last_path)
    if resume_from is not None:
        state = torch.load(resume_from, map_location="cpu", weights_only=False)
        if state.get("fingerprint") != cfg.fingerprint() and not cfg.force_resume:
            raise RuntimeError(
                f"Refusing to resume {resume_from}: its config fingerprint "
                f"({state.get('fingerprint')}) differs from this run's ({cfg.fingerprint()}). A "
                "changed stage, schedule, loss or augmentation strength makes the saved optimiser and "
                "schedule meaningless. Use a fresh --out, or pass --force-resume to override.")
        core.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        restored = ["model", "EMA", "optimiser"]
        if amp and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
            restored.append("GradScaler")
        history, best = state["history"], state["best"]
        bad_epochs, start_epoch = state["bad_epochs"], state["epoch"] + 1
        global_update = state["global_update"]
        elapsed_before = state.get("elapsed", 0.0)
        rng_states = state.get("rng")
        if rng_states and len(rng_states) == world:
            set_rng_state(rng_states[rank])
            restored.append("RNG states")
        restored += [f"LR schedule at update {global_update}", f"history ({len(history)} epochs)",
                     f"best {best['score']:.4f} @ epoch {best['epoch']} ({best['source']})",
                     f"early-stop counter {bad_epochs}/{cfg.patience}"]
        mode = f"RESUMED from epoch {start_epoch - 1}"
        detail = "restored: " + ", ".join(restored) + f" | continuing at epoch {start_epoch} of {cfg.epochs}"
        if is_main:
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                if history:
                    writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                    writer.writeheader()
                    writer.writerows(history)
    elif cfg.init_from:
        report = load_weights_only(core, ema, cfg.init_from)
        mode = "WEIGHTS-ONLY RESUME"
        detail = (f"from {cfg.init_from} | restored: {', '.join(report['restored'])} (EMA re-seeded "
                  f"from them) | fresh: optimiser, LR schedule, GradScaler, RNG, epoch counter"
                  + (f", {', '.join(report['fresh'])} (random init)" if report["fresh"] else ""))
    else:
        mode = "FRESH START"
        source = pretrained_source(cfg)
        detail = (f"stage {cfg.stage} | encoder {cfg.encoder} "
                  + (f"initialised from pretrained '{source}'" if source else "from RANDOM weights")
                  + " | nothing restored")
    if is_main:
        logger.info("%s | %s", mode, detail)
        logger.info(
            "device=%s world=%d amp=%s | stage %s heads %s | %d params | train %d / val %d images "
            "(%d supervised in train) | %d steps/epoch, batch %d x %d GPU x accum %d | fingerprint %s",
            device, world, amp, cfg.stage, list(spec.heads), count_parameters(core), len(train_ds),
            len(val_ds), int(train_ds.sup.sum()), steps_per_epoch, cfg.batch_size, world,
            cfg.grad_accum, cfg.fingerprint())

    started = time.time()
    epoch_times: list[float] = []
    stop_requested = False

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_start = time.time()
        sampler.set_epoch(epoch)
        train_ds.set_epoch(epoch)
        model.train()
        sums = {k: torch.zeros((), device=device) for k in (
            "total", "lung_bce", "lung_dice_loss", "opacity_bce", "opacity_dice_loss",
            "n_supervised")}
        grad_norms: list[float] = []
        skipped = 0
        optimizer.zero_grad(set_to_none=True)
        lr_now = cfg.lr

        for step, batch in enumerate(loader):
            if step >= steps_per_epoch:
                break
            full = batch["full"].to(device, non_blocking=True)
            lung_t = batch["lung"].to(device, non_blocking=True)
            boundary = (step + 1) % cfg.grad_accum == 0
            lr_now = learning_rate(global_update, total_updates, warmup_updates, cfg.lr,
                                   cfg.min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = lr_now * group.get("lr_mult", 1.0)

            crop_n = opacity_t = valid = None
            if has_opacity:
                sup = batch["sup"].to(device)
                idx = sup.nonzero().squeeze(1)
                if idx.numel():
                    crop_n = normalise_u8(batch["crop"].to(device)[idx])
                    opacity_t = batch["opacity"].to(device)[idx]
                    valid = torch.ones(idx.numel(), device=device)
                else:
                    # The sampler guarantees supervised samples; this keeps the graph (and
                    # DDP's reducer) complete if one is ever missing, at zero loss weight.
                    crop_n = normalise_u8(full[:1])
                    opacity_t = torch.zeros_like(lung_t[:1])
                    valid = torch.zeros(1, device=device)

            sync = contextlib.nullcontext()
            if world > 1 and not boundary:
                sync = model.no_sync()  # type: ignore[union-attr]
            with sync:
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    lung_logits, op_logits = model(normalise_u8(full), crop_n)
                loss, parts = criterion(lung_logits.float(), lung_t,
                                        None if op_logits is None else op_logits.float(),
                                        opacity_t, valid)
                if not _all_true(bool(torch.isfinite(loss)), device):
                    skipped += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss / cfg.grad_accum).backward()

            for key in sums:
                sums[key] += parts[key]

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
                logger.info("  ep %03d step %4d/%d loss %.4f lr %.2e%s", epoch, step + 1,
                            steps_per_epoch, float(sums["total"]) / max(1, step + 1 - skipped),
                            lr_now, f" gpu {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GB"
                            if device.type == "cuda" else "")

        done_steps = max(1, steps_per_epoch - skipped)
        stacked = torch.stack(list(sums.values()))
        if _is_dist():
            dist.all_reduce(stacked)
            stacked /= world
        train = dict(zip(sums, (stacked / done_steps).tolist(), strict=True))

        eval_kwargs = {"device": device, "amp": amp, "batch_size": cfg.batch_size * 2,
                       "workers": cfg.workers, "criterion": criterion, "has_opacity": has_opacity}
        raw = evaluate(core, val_ds, **eval_kwargs)
        smooth = evaluate(ema.module, val_ds, **eval_kwargs)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        elapsed = elapsed_before + (time.time() - started)

        stop = False
        if is_main:
            raw_score, raw_op = selection_score(raw, cfg, has_opacity)
            ema_score, ema_op = selection_score(smooth, cfg, has_opacity)
            winner, score = ("ema", ema_score) if ema_score >= raw_score else ("raw", raw_score)
            improved = score > best["score"] + cfg.min_delta
            if improved:
                best = {"score": score, "epoch": epoch, "source": winner}
                bad_epochs = 0
                if not cfg.is_probe:
                    _save_best(cfg, spec, core, ema, winner, epoch, raw, smooth, world, split)
            else:
                bad_epochs += 1

            row = {
                "epoch": epoch, "lr": lr_now, "train_loss": train["total"],
                "train_lung_bce": train["lung_bce"], "train_lung_dice_loss": train["lung_dice_loss"],
                "train_opacity_bce": train["opacity_bce"],
                "train_opacity_dice_loss": train["opacity_dice_loss"],
                "train_supervised_per_step": train["n_supervised"],
                "val_loss_raw": raw["loss"], "val_loss_ema": smooth["loss"],
                "val_lung_dice_raw": raw["lung_dice"], "val_lung_dice_ema": smooth["lung_dice"],
                "val_lung_iou_ema": smooth["lung_iou"],
                "val_opacity_pos_dice_raw": raw["opacity_pos_dice"],
                "val_opacity_pos_dice_ema": smooth["opacity_pos_dice"],
                "val_normal_fpr_raw": raw["normal_fpr"], "val_normal_fpr_ema": smooth["normal_fpr"],
                "val_normal_frac_ema": smooth["normal_mean_fraction"],
                "val_opacity_score_raw": raw_op, "val_opacity_score_ema": ema_op,
                "val_score_raw": raw_score, "val_score_ema": ema_score, "winner": winner,
                "grad_norm": float(np.mean(grad_norms)) if grad_norms else float("nan"),
                "skipped_steps": skipped, "epoch_seconds": epoch_time,
                "best_score": best["score"], "improved": int(improved),
            }
            history.append(row)
            _append_csv(csv_path, row)
            with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as handle:
                json.dump(plain(history), handle, indent=1)

            def fmt(value: float | None) -> str:
                return "  n/a" if value is None else f"{value:.3f}"

            logger.info(
                "EPOCH %03d/%d | train %.4f | val raw %.4f ema %.4f (%s) | lung %.3f | opacity pos "
                "%s | Normal FPR %s | lr %.2e | gn %.2f | %.1f min | best %.4f @%d (%s) | patience "
                "%d/%d%s", epoch, cfg.epochs, train["total"], raw_score, ema_score, winner,
                smooth["lung_dice"], fmt(smooth["opacity_pos_dice"]), fmt(smooth["normal_fpr"]),
                lr_now, row["grad_norm"], epoch_time / 60, best["score"], best["epoch"],
                best["source"], bad_epochs, cfg.patience, "  *best*" if improved else "")
            if skipped:
                logger.warning("epoch %d skipped %d non-finite steps", epoch, skipped)

            if bad_epochs >= cfg.patience:
                logger.info("EARLY STOP: no improvement > %.0e for %d epochs", cfg.min_delta, cfg.patience)
                stop = True
            budget = cfg.time_budget_hours * 3600.0
            if not stop and elapsed + 1.15 * float(np.mean(epoch_times[-3:])) > budget:
                logger.warning("TIME BUDGET: %.2f h used of %.2f h; stopping so the final test, plots "
                               "and ZIP still run", elapsed / 3600, cfg.time_budget_hours)
                stop = True

        rng_all = _gather(rng_state())
        if is_main and not cfg.is_probe:
            _save_last(last_path, core, ema, optimizer, scaler, epoch, history, best, bad_epochs,
                       global_update, cfg, elapsed, rng_all)
        stop_requested = _broadcast_flag(stop, device)
        if stop_requested:
            break

    if _is_dist():
        dist.barrier()
    if is_main:
        with open(os.path.join(out_dir, "training_done.json"), "w", encoding="utf-8") as handle:
            json.dump(plain({
                "best": best, "epochs_run": len(history), "stopped_early": bad_epochs >= cfg.patience,
                "stopped_by_time_budget": stop_requested and bad_epochs < cfg.patience,
                "elapsed_hours": (elapsed_before + time.time() - started) / 3600,
                "stage": cfg.stage, "fingerprint": cfg.fingerprint(),
            }), handle, indent=1)
        logger.info("training finished: best %.4f at epoch %d (%s weights)", best["score"],
                    best["epoch"], best["source"])


# ----------------------------------------------------------------------------- saving


def _append_csv(path: str, row: dict[str, Any]) -> None:
    new = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def default_postprocess() -> dict[str, Any]:
    return {"lung": LungPost().to_dict(), "opacity": OpacityPost().to_dict(),
            "tuned_on_validation": False}


def _save_best(
    cfg: TrainConfig, spec: SegSpec, core: LungSegNet, ema: EMA, winner: str, epoch: int,
    raw: dict[str, Any], smooth: dict[str, Any], world: int, split: splits_mod.SplitResult,
) -> None:
    source = ema.module if winner == "ema" else core
    chosen = smooth if winner == "ema" else raw
    post = default_postprocess()
    if "opacity" not in spec.heads:
        post["opacity"] = None
    payload = build_serving_checkpoint(
        source.state_dict() if False else source.canonical_state_dict(), spec,  # canonical names
        heads_trained=cfg.heads(), postprocess=post,
        metrics={"val_during_training": chosen},
        training={"epoch": epoch, "weights": winner, "stage": cfg.stage, "seed": cfg.seed,
                  "gpus": world, "pretrained": cfg.pretrained or None,
                  "split_counts": {n: len(split.part(n)) for n in ("train", "val", "test")},
                  "config_fingerprint": cfg.fingerprint(), "phase": "best-during-training"},
    )
    save_serving_checkpoint(payload, os.path.join(cfg.out_dir, "best", CHECKPOINT_NAME))


def _save_last(
    path: str, core: nn.Module, ema: EMA, optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler, epoch: int, history: list[dict[str, Any]], best: dict[str, Any],
    bad_epochs: int, global_update: int, cfg: TrainConfig, elapsed: float, rng_all: list[Any],
) -> None:
    tmp = path + ".tmp"
    torch.save({
        "model": core.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None, "epoch": epoch,
        "history": history, "best": best, "bad_epochs": bad_epochs, "global_update": global_update,
        "fingerprint": cfg.fingerprint(), "elapsed": elapsed, "rng": rng_all, "stage": cfg.stage,
    }, tmp)
    os.replace(tmp, path)
