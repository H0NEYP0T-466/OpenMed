"""LungV2S — 4-class chest X-ray classifier, Kaggle training entrypoint.

    python train_lungs_v2s.py --data-root /kaggle/input --output-dir /kaggle/working/lung_v2s_run
    python train_lungs_v2s.py --audit-only --data-root /path/to/dataset

Backbone ``tf_efficientnetv2_s.in21k_ft_in1k`` (timm), 300 px training, 4-class
softmax over COVID / Lung_Opacity / Normal / Viral Pneumonia, trained from the
COVID-19 Radiography Database.

What this run does, in order:

1. Discover the dataset root under ``/kaggle/input`` (bounded walk, depth 4).
2. Hash every image (SHA-256 + pHash), cluster duplicates with union-find and
   write the hygiene report. ``--audit-only`` stops here.
3. Split by duplicate cluster, stratified per class, 70/15/15, and assert that
   no cluster spans two splits.
4. Train with class-weighted CE (inverse frequency, mean 1, label smoothing
   0.1), AdamW + OneCycle, MixUp/CutMix, EMA, AMP, Grad-CAM and full metrics.
5. Pick ``infer_size`` on validation (300 vs 384, with and without TTA), run the
   final test with 4-view TTA, write plots/metrics/sidecar/manifest and zip the
   whole run directory.

Resume is explicit and loud: ``RESUMED from epoch N``, ``WEIGHTS-ONLY RESUME``
or ``FRESH START``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
import zipfile
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

try:  # package import: app.organs.lungs.classification.kaggle.train_lungs_v2s
    from . import hygiene, plots
    from .config import (
        CLASS_NAMES,
        IMAGENET_MEAN,
        IMAGENET_STD,
        KAGGLE_WORKING,
        MODEL_TAG,
        NUM_CLASSES,
        LungConfig,
        class_dirs,
        resolve_dataset_root,
    )
except ImportError:  # direct execution: python train_lungs_v2s.py
    import hygiene
    import plots
    from config import (
        CLASS_NAMES,
        IMAGENET_MEAN,
        IMAGENET_STD,
        KAGGLE_WORKING,
        MODEL_TAG,
        NUM_CLASSES,
        LungConfig,
        class_dirs,
        resolve_dataset_root,
    )

# The dataset module imports torch, timm and torchvision. It is imported
# defensively so `--audit-only` still starts on a machine with none of them —
# the whole point of that mode is auditing the data before a GPU run exists.
_DATASET_IMPORT_ERROR: Optional[BaseException] = None
try:
    if __package__:
        from .dataset import (
            LungDataset,
            build_eval_transform,
            build_train_transform,
            compute_class_weights,
            grouped_stratified_split,
            load_rgb,
            make_loader,
            manifest_rows,
            preview_augmentations,
            split_counts,
        )
    else:
        raise ImportError("direct execution")
except ImportError:
    try:
        from dataset import (  # type: ignore[no-redef]
            LungDataset,
            build_eval_transform,
            build_train_transform,
            compute_class_weights,
            grouped_stratified_split,
            load_rgb,
            make_loader,
            manifest_rows,
            preview_augmentations,
            split_counts,
        )
    except ImportError as exc:  # torch-free audit-only run
        _DATASET_IMPORT_ERROR = exc

LOGGER = logging.getLogger("lung_v2s")

HISTORY_KEYS = (
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss_raw",
    "val_acc_raw",
    "val_f1_raw",
    "val_loss_ema",
    "val_acc_ema",
    "val_f1_ema",
    "val_loss",
    "val_acc",
    "val_macro_f1",
    "lr",
    "mixup",
    "epoch_seconds",
)

HISTORY_HEADER = [
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss_raw",
    "val_acc_raw",
    "val_f1_raw",
    "val_loss_ema",
    "val_acc_ema",
    "val_f1_ema",
    "val_loss_best",
    "val_f1_best",
    "lr",
    "mixup",
    "epoch_seconds",
]


# ─────────────────────────────────────────────────────────────────────────────
# Plumbing
# ─────────────────────────────────────────────────────────────────────────────


def setup_logging(log_path: Path) -> None:
    """stdout + rotating file, format ``%(asctime)s %(levelname)s %(message)s``."""
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for handler in list(LOGGER.handlers):
        LOGGER.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        str(log_path), maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # audit-only on a torch-free laptop
        return


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LungV2S (EfficientNetV2-S) on the chest X-ray 4-class corpus"
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("OPENMED_LUNGS_DATASET", "/kaggle/input"),
        help="Where to look for the dataset (walked to depth 4). Default /kaggle/input",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OPENMED_LUNGS_OUTPUT", os.path.join(KAGGLE_WORKING, "lung_v2s_run")),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run the hygiene audit on CPU and exit without training",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume", dest="resume", action="store_true", default=True,
        help="Resume from checkpoints/lung_v2s_resume.pth when present (default)",
    )
    resume_group.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Ignore existing checkpoints and start from scratch",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    return args


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────


def prepare_data(
    cfg: LungConfig, dataset_root: str, hygiene_dir: Path
) -> tuple[list[hygiene.ImageRecord], dict[int, list[int]], Any, dict[str, Any]]:
    """Scan, cluster, split and audit. Returns records, clusters, plan, report."""
    LOGGER.info("Scanning %s", dataset_root)
    directories = class_dirs(dataset_root)
    for name, path in directories.items():
        LOGGER.info("  %-16s %s", name, path)

    started = time.time()
    records = hygiene.scan(
        directories, CLASS_NAMES, dataset_root, workers=cfg.hash_workers
    )
    hygiene.assign_sources(records, dataset_root)
    cluster_members, cluster_stats = hygiene.cluster_records(records, cfg.phash_hamming)
    LOGGER.info(
        "Hashed %d images in %.1fs | readable=%d unreadable=%d clusters=%d largest=%d",
        len(records),
        time.time() - started,
        cluster_stats["readable"],
        cluster_stats["unreadable"],
        cluster_stats["clusters"],
        cluster_stats["largest_cluster"],
    )

    plan = grouped_stratified_split(
        records,
        cluster_members,
        CLASS_NAMES,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        seed=cfg.seed,
    )
    LOGGER.info(
        "Split by cluster: train=%d val=%d test=%d (70/15/15, stratified per class)",
        len(plan.train),
        len(plan.val),
        len(plan.test),
    )
    for class_name, counts in plan.per_class.items():
        LOGGER.info(
            "  %-16s train=%5d val=%5d test=%5d",
            class_name,
            counts["train"],
            counts["val"],
            counts["test"],
        )

    report = hygiene.build_report(
        records,
        dataset_root,
        cluster_members,
        hamming_max=cfg.phash_hamming,
        split_of=plan.split_of,
        cluster_stats=cluster_stats,
    )
    paths = hygiene.write_report(report, hygiene_dir)
    hygiene.write_records_csv(records, hygiene_dir / "report.csv", plan.split_of)
    LOGGER.info("Hygiene report: %s", paths["json"])
    for line in hygiene.summary_lines(report):
        LOGGER.info("  %s", line)
    for reason in report["verdict_reasons"]:
        LOGGER.info("  %s", reason)
    if report["verdict"] == "FAIL":
        raise RuntimeError(
            "Hygiene audit returned FAIL; refusing to train on a leaking split. "
            f"See {paths['md']}"
        )
    return records, cluster_members, plan, report


# ─────────────────────────────────────────────────────────────────────────────
# Model, loss, optimisation
# ─────────────────────────────────────────────────────────────────────────────


def build_model(cfg: LungConfig, device: Any) -> Any:
    import timm

    try:
        model = timm.create_model(
            cfg.model_tag,
            pretrained=True,
            num_classes=cfg.num_classes,
            drop_path_rate=cfg.drop_path_rate,
            drop_rate=cfg.head_dropout,
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with actionable text
        raise RuntimeError(
            f"Could not build {cfg.model_tag}. On Kaggle enable Internet so timm can "
            "download the ImageNet weights, or attach them as a dataset and set "
            f"HF_HOME/TIMM cache. Original error: {type(exc).__name__}: {exc}"
        ) from exc
    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(
        "Model %s | %d classes | %.2fM trainable params | drop_path=%.2f head_drop=%.2f",
        cfg.model_tag,
        cfg.num_classes,
        trainable / 1e6,
        cfg.drop_path_rate,
        cfg.head_dropout,
    )
    return model


def build_optimizer(model: Any, cfg: LungConfig) -> Any:
    import torch

    decay: list[Any] = []
    no_decay: list[Any] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    LOGGER.info(
        "AdamW lr=%g wd=%g | %d decayed tensors, %d exempt (norm+bias)",
        cfg.lr,
        cfg.weight_decay,
        len(decay),
        len(no_decay),
    )
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )


def build_scheduler(optimizer: Any, cfg: LungConfig, steps_per_epoch: int) -> Any:
    import torch

    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=max(steps_per_epoch, 1),
        pct_start=cfg.pct_start,
    )


def build_mixup(cfg: LungConfig) -> Any:
    from timm.data import Mixup

    return Mixup(
        mixup_alpha=cfg.mixup_alpha,
        cutmix_alpha=cfg.cutmix_alpha,
        prob=cfg.mixup_prob,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.0,  # the criterion owns smoothing; avoid smoothing twice
        num_classes=cfg.num_classes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    from sklearn.metrics import f1_score

    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        )
    )


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: Sequence[str]
) -> dict[str, Any]:
    """Accuracy, macro-F1, macro AUC (OvR), per-class P/R/F1/support, confusion."""
    from sklearn.metrics import (
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = y_prob.argmax(axis=1)
    num_classes = len(class_names)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(class_names)
    }

    macro_auc: Optional[float]
    try:
        macro_auc = float(
            roc_auc_score(
                y_true,
                y_prob,
                labels=list(range(num_classes)),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        macro_auc = None

    return {
        "accuracy": float((y_pred == y_true).mean()) if len(y_true) else 0.0,
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "macro_auc_ovr": macro_auc,
        "per_class": per_class,
        "confusion": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))),
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Train / evaluate
# ─────────────────────────────────────────────────────────────────────────────


def train_one_epoch(
    model: Any,
    ema: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    device: Any,
    epoch: int,
    mixup_fn: Optional[Any],
    use_amp: bool,
    grad_clip: float,
) -> tuple[float, float]:
    import torch

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    started = time.time()

    for step, (images, labels, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if mixup_fn is not None and images.size(0) > 1:
                mixed, targets = mixup_fn(images, labels)
                logits = model(mixed)
                loss = criterion(logits, targets)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        batch = images.size(0)
        running_loss += float(loss.item()) * batch
        correct += int(logits.detach().argmax(1).eq(labels).sum().item())
        total += batch

        if step % 50 == 0:
            LOGGER.info(
                "  epoch %03d step %04d/%04d loss=%.4f acc=%.4f lr=%.2e (%.0fs)",
                epoch,
                step,
                len(loader),
                running_loss / max(total, 1),
                correct / max(total, 1),
                optimizer.param_groups[0]["lr"],
                time.time() - started,
            )

    return running_loss / max(total, 1), correct / max(total, 1)


def evaluate(
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    use_amp: bool,
    num_classes: int,
) -> dict[str, Any]:
    import torch

    model.eval()
    running_loss = 0.0
    total = 0
    probability_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            running_loss += float(loss.item()) * images.size(0)
            total += images.size(0)
            probability_chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            label_chunks.append(labels.cpu().numpy())

    if not probability_chunks:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "y_true": np.zeros(0, dtype=np.int64),
            "y_prob": np.zeros((0, num_classes), dtype=np.float64),
        }

    y_prob = np.vstack(probability_chunks)
    y_true = np.concatenate(label_chunks).astype(np.int64)
    y_pred = y_prob.argmax(axis=1)
    return {
        "loss": running_loss / max(total, 1),
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "y_true": y_true,
        "y_prob": y_prob,
    }


def predict_probs(
    model: Any,
    records: Sequence[hygiene.ImageRecord],
    indices: Sequence[int],
    device: Any,
    size: int,
    hflip: bool,
    batch_size: int,
    num_workers: int,
    seed: int,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Softmax probabilities at one resolution, optionally averaged over hflip."""
    import torch

    dataset = LungDataset(records, indices, build_eval_transform(size))
    loader = make_loader(dataset, batch_size, False, num_workers, seed, drop_last=False)
    views = [False, True] if hflip else [False]

    model.eval()
    view_probabilities: list[np.ndarray] = []
    labels_out: Optional[np.ndarray] = None
    with torch.no_grad():
        for view_index, flip in enumerate(views):
            chunks: list[np.ndarray] = []
            label_chunks: list[np.ndarray] = []
            for images, labels, _ in loader:
                images = images.to(device, non_blocking=True)
                if flip:
                    images = torch.flip(images, dims=[3])
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(images)
                chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
                label_chunks.append(labels.numpy())
            if chunks:
                view_probabilities.append(np.concatenate(chunks))
            if view_index == 0:
                labels_out = np.concatenate(label_chunks) if label_chunks else np.zeros(0, np.int64)

    if not view_probabilities:
        return np.zeros((0, len(CLASS_NAMES)), dtype=np.float64), np.zeros(0, dtype=np.int64)
    probabilities = np.mean(np.stack(view_probabilities, axis=0), axis=0)
    return probabilities, labels_out if labels_out is not None else np.zeros(0, np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────


def gradcam_overlay(
    model: Any, tensor: Any, target_class: int, image_rgb: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """Grad-CAM on ``conv_head`` blended onto the RGB image. No cv2 needed."""
    import torch
    from matplotlib import cm
    from PIL import Image as PILImage

    activations: dict[str, Any] = {}

    def _capture(module: Any, inputs: Any, output: Any) -> None:
        activations["value"] = output

    handle = model.conv_head.register_forward_hook(_capture)
    was_training = model.training
    model.eval()
    try:
        model.zero_grad(set_to_none=True)
        probe = tensor.clone().detach().requires_grad_(True)
        logits = model(probe)
        value = activations.get("value")
        if value is None:
            raise RuntimeError("Grad-CAM hook captured no activation from conv_head")
        gradient = torch.autograd.grad(logits[0, int(target_class)], value)[0]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * value).sum(dim=1))[0]
        cam = cam.detach().float().cpu().numpy()
        cam -= float(cam.min())
        peak = float(cam.max())
        if peak > 0:
            cam /= peak
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()

    height, width = image_rgb.shape[:2]
    resized = np.asarray(
        PILImage.fromarray((cam * 255.0).astype(np.uint8)).resize((width, height), PILImage.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    heat = cm.jet(resized)[..., :3]
    base = image_rgb.astype(np.float32) / 255.0
    blended = np.clip(alpha * heat + (1.0 - alpha) * base, 0.0, 1.0)
    return (blended * 255.0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoints
# ─────────────────────────────────────────────────────────────────────────────


def _snapshot(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def save_weight_checkpoint(
    path: Path,
    state: dict[str, Any],
    epoch: int,
    kind: str,
    metrics: dict[str, Any],
    cfg: LungConfig,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": _snapshot(state),
            "epoch": int(epoch),
            "kind": kind,
            "metrics": _jsonable(metrics),
            "class_names": list(CLASS_NAMES),
            "model_tag": MODEL_TAG,
            "input_size": cfg.train_size,
        },
        path,
    )


def save_resume_checkpoint(
    path: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    ema: Any,
    epoch: int,
    best_metric: float,
    stale: int,
    history: dict[str, list[Any]],
    cfg: LungConfig,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": _snapshot(model.state_dict()),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": _snapshot(ema.module.state_dict()) if ema is not None else None,
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "epochs_without_improvement": int(stale),
            "history": history,
            "config": cfg.to_dict(),
        },
        path,
    )


def load_torch(path: Path, device: Any) -> dict[str, Any]:
    import torch

    # weights_only=False: the payload carries plain python metadata (config,
    # history, class names), not just tensors. These files are written by this
    # script and read back by it.
    return torch.load(str(path), map_location=device, weights_only=False)


# ─────────────────────────────────────────────────────────────────────────────
# Packaging
# ─────────────────────────────────────────────────────────────────────────────


def zip_run_dir(run_dir: Path, zip_path: Path) -> tuple[str, int, int]:
    """Zip everything under ``run_dir``. Returns (path, file count, bytes)."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    files = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for root, _dirs, names in os.walk(run_dir):
            for name in sorted(names):
                full = Path(root) / name
                if full.resolve() == zip_path.resolve():
                    continue
                archive.write(full, os.path.relpath(full, run_dir))
                files += 1
    return str(zip_path), files, zip_path.stat().st_size


def write_history_csv(path: Path, history: dict[str, list[Any]]) -> None:
    """Rewrite ``history.csv`` from the in-memory history (one row per epoch).

    Rewriting rather than appending keeps resumed runs correct: the restored
    history already holds every earlier row, so the file always matches memory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HISTORY_HEADER)
        for index in range(len(history["epoch"])):
            writer.writerow(
                [
                    history["epoch"][index],
                    f"{history['train_loss'][index]:.6f}",
                    f"{history['train_acc'][index]:.6f}",
                    f"{history['val_loss_raw'][index]:.6f}",
                    f"{history['val_acc_raw'][index]:.6f}",
                    f"{history['val_f1_raw'][index]:.6f}",
                    f"{history['val_loss_ema'][index]:.6f}",
                    f"{history['val_acc_ema'][index]:.6f}",
                    f"{history['val_f1_ema'][index]:.6f}",
                    f"{history['val_loss'][index]:.6f}",
                    f"{history['val_macro_f1'][index]:.6f}",
                    f"{history['lr'][index]:.8f}",
                    int(history["mixup"][index]),
                    f"{history['epoch_seconds'][index]:.2f}",
                ]
            )


def write_split_manifest(path: Path, rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "class", "sha256", "phash", "cluster_id", "split"])
        writer.writerows(rows)


def resolve_zip_path(run_dir: Path) -> Path:
    """``/kaggle/working/output.zip`` on Kaggle, next to the run dir elsewhere."""
    working = Path(KAGGLE_WORKING)
    try:
        if working.is_dir() and os.access(working, os.W_OK) and run_dir.resolve().is_relative_to(working):
            return working / "output.zip"
    except (OSError, ValueError):
        pass
    return run_dir.parent / "output.zip"


# ─────────────────────────────────────────────────────────────────────────────
# Audit-only mode
# ─────────────────────────────────────────────────────────────────────────────


def run_audit_only(cfg: LungConfig, dataset_root: str, hygiene_dir: Path) -> int:
    LOGGER.info("=" * 72)
    LOGGER.info("  LungV2S hygiene audit (no training)")
    LOGGER.info("  data_root=%s", dataset_root)
    LOGGER.info("  report_dir=%s", hygiene_dir)
    LOGGER.info("=" * 72)

    directories = class_dirs(dataset_root)
    for name, path in directories.items():
        LOGGER.info("  %-16s %s", name, path)

    started = time.time()
    records = hygiene.scan(directories, CLASS_NAMES, dataset_root, workers=cfg.hash_workers)
    hygiene.assign_sources(records, dataset_root)
    cluster_members, cluster_stats = hygiene.cluster_records(records, cfg.phash_hamming)
    LOGGER.info("Hashed %d images in %.1fs", len(records), time.time() - started)

    split_of: Optional[dict[int, str]] = None
    if _DATASET_IMPORT_ERROR is not None:
        LOGGER.warning(
            "torch is not installed (%s): split/leakage audit skipped, duplicate "
            "audit only",
            _DATASET_IMPORT_ERROR,
        )
    else:
        try:
            plan = grouped_stratified_split(
                records,
                cluster_members,
                CLASS_NAMES,
                val_frac=cfg.val_frac,
                test_frac=cfg.test_frac,
                seed=cfg.seed,
            )
            split_of = plan.split_of
        except RuntimeError as exc:
            # e.g. too few independent clusters to fill three splits on a highly
            # duplicated corpus. Degrade to a duplicate-only audit rather than
            # crashing the one mode that is supposed to be safe to run anywhere.
            LOGGER.warning("split/leakage audit skipped: %s", exc)

    report = hygiene.build_report(
        records,
        dataset_root,
        cluster_members,
        hamming_max=cfg.phash_hamming,
        split_of=split_of,
        cluster_stats=cluster_stats,
    )
    paths = hygiene.write_report(report, hygiene_dir)
    hygiene.write_records_csv(records, hygiene_dir / "report.csv", split_of)

    LOGGER.info("-" * 72)
    for line in hygiene.summary_lines(report):
        LOGGER.info("  %s", line)
    for reason in report["verdict_reasons"]:
        LOGGER.info("  %s", reason)
    LOGGER.info("  json: %s", paths["json"])
    LOGGER.info("  csv : %s", hygiene_dir / "report.csv")
    LOGGER.info("  md  : %s", paths["md"])
    LOGGER.info("-" * 72)
    return 0 if report["verdict"] in ("PASS", "WARN") else 1


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = LungConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    run_dir = Path(args.output_dir).expanduser().resolve()
    checkpoints_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    hygiene_dir = run_dir / "hygiene"
    logs_dir = run_dir / "logs"
    for directory in (run_dir, checkpoints_dir, plots_dir, hygiene_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(run_dir)

    setup_logging(logs_dir / "train.log")
    set_seed(cfg.seed)

    dataset_root = resolve_dataset_root(args.data_root)
    LOGGER.info("Dataset root: %s", dataset_root)

    if args.audit_only:
        return run_audit_only(cfg, dataset_root, hygiene_dir)

    import torch
    from timm.utils import ModelEmaV2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.amp and device.type == "cuda")

    LOGGER.info("=" * 72)
    LOGGER.info("  LungV2S training run")
    LOGGER.info("  device=%s  seed=%d  batch=%d  epochs=%d  amp=%s", device, cfg.seed, cfg.batch_size, cfg.epochs, use_amp)
    if device.type == "cuda":
        LOGGER.info(
            "  gpu=%s (%.1f GB)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    LOGGER.info("  data_root=%s", dataset_root)
    LOGGER.info("  output_dir=%s", run_dir)
    LOGGER.info("=" * 72)

    # ── data ────────────────────────────────────────────────────────────
    records, cluster_members, plan, report = prepare_data(cfg, dataset_root, hygiene_dir)

    class_weight_vector = compute_class_weights(plan.train, records, NUM_CLASSES)
    LOGGER.info(
        "Class weights (train, mean 1): %s",
        {name: round(float(class_weight_vector[i]), 4) for i, name in enumerate(CLASS_NAMES)},
    )
    per_split = split_counts(plan, records, CLASS_NAMES)
    write_split_manifest(run_dir / "split_manifest.csv", manifest_rows(records, plan))

    # ── dataset figures ─────────────────────────────────────────────────
    counts = {name: int(sum(per_split[split][name] for split in per_split)) for name in CLASS_NAMES}
    plots.plot_class_distribution(counts, str(plots_dir / "class_distribution.png"))
    plots.plot_split_distribution(per_split, CLASS_NAMES, str(plots_dir / "split_distribution.png"))
    plots.plot_duplicate_cluster_histogram(
        [len(members) for members in cluster_members.values()],
        str(plots_dir / "duplicate_cluster_histogram.png"),
    )
    plots.plot_image_size_histogram(
        [(record.width, record.height) for record in records if record.width],
        str(plots_dir / "image_size_histogram.png"),
    )

    samples: dict[str, list[np.ndarray]] = {}
    for class_index, name in enumerate(CLASS_NAMES):
        picks = [index for index in plan.test if records[index].class_index == class_index][:4]
        samples[name] = [np.asarray(load_rgb(records[index].path)) for index in picks]
    plots.plot_sample_grid_per_class(samples, str(plots_dir / "sample_grid_per_class.png"))

    preview_transform = build_train_transform(cfg.train_size, cfg)
    minority_index = CLASS_NAMES.index(cfg.minority_class)
    preview_pool = [index for index in plan.train if records[index].class_index == minority_index]
    preview_index = preview_pool[0] if preview_pool else plan.train[0]
    plots.plot_augmentation_preview(
        preview_augmentations(records, preview_index, preview_transform, count=8, seed=cfg.seed),
        str(plots_dir / "augmentation_preview.png"),
        title=f"Augmentation preview — {records[preview_index].class_name}",
    )

    # ── loaders ─────────────────────────────────────────────────────────
    train_dataset = LungDataset(
        records,
        plan.train,
        build_train_transform(cfg.train_size, cfg),
        minority_class_index=minority_index,
        minority_extra_draw_prob=cfg.minority_extra_draw_prob,
        minority_transform=build_train_transform(
            cfg.train_size, cfg, rand_augment=cfg.minority_rand_augment
        ),
    )
    val_dataset = LungDataset(records, plan.val, build_eval_transform(cfg.train_size))

    train_loader = make_loader(train_dataset, cfg.batch_size, True, cfg.num_workers, cfg.seed)
    val_loader = make_loader(val_dataset, cfg.batch_size, False, cfg.num_workers, cfg.seed, drop_last=False)
    steps_per_epoch = max(len(train_loader), 1)
    LOGGER.info(
        "Loaders: %d train / %d val / %d test images, %d steps per epoch",
        len(train_dataset),
        len(val_dataset),
        len(plan.test),
        steps_per_epoch,
    )

    # ── model + optimisation ────────────────────────────────────────────
    model = build_model(cfg, device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weight_vector, dtype=torch.float32, device=device),
        label_smoothing=cfg.label_smoothing,
    )
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    mixup_fn = build_mixup(cfg)
    ema: Optional[Any] = ModelEmaV2(model, decay=cfg.ema_decay)

    # ── resume ──────────────────────────────────────────────────────────
    best_path = checkpoints_dir / "lung_v2s_best.pth"
    last_path = checkpoints_dir / "lung_v2s_last.pth"
    resume_path = checkpoints_dir / "lung_v2s_resume.pth"

    history: dict[str, list[Any]] = {key: [] for key in HISTORY_KEYS}
    best_metric = float("-inf")
    stale_epochs = 0
    start_epoch = 1

    if args.resume and resume_path.exists():
        payload = load_torch(resume_path, device)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        if payload.get("ema") is not None and ema is not None:
            ema.module.load_state_dict(payload["ema"])
        best_metric = float(payload.get("best_metric", float("-inf")))
        stale_epochs = int(payload.get("epochs_without_improvement", 0))
        restored = payload.get("history") or {}
        history = {key: list(restored.get(key, [])) for key in HISTORY_KEYS}
        # A payload written mid-epoch (or by an older revision) can leave the
        # per-epoch columns at different lengths; keep the common prefix.
        lengths = {len(values) for values in history.values()}
        if len(lengths) > 1:
            keep = min(lengths)
            LOGGER.warning("History columns disagree in length; truncating to %d rows", keep)
            history = {key: values[:keep] for key, values in history.items()}
        start_epoch = int(payload["epoch"]) + 1
        LOGGER.info("RESUMED from epoch %d", int(payload["epoch"]))
        LOGGER.info(
            "  restored optimizer/scheduler/EMA, best_metric=%.4f, stale=%d, %d history rows",
            best_metric,
            stale_epochs,
            len(history["epoch"]),
        )
    elif args.resume and last_path.exists():
        payload = load_torch(last_path, device)
        model.load_state_dict(payload["state_dict"])
        start_epoch = 1
        LOGGER.info("WEIGHTS-ONLY RESUME")
        LOGGER.info(
            "  loaded %s (epoch %s); optimizer/scheduler/EMA start fresh",
            last_path.name,
            payload.get("epoch"),
        )
    else:
        if last_path.exists() or resume_path.exists():
            LOGGER.warning(
                "--no-resume: existing checkpoints in %s will be overwritten", checkpoints_dir
            )
        LOGGER.info("FRESH START")

    if start_epoch > cfg.epochs:
        LOGGER.info(
            "Checkpoint is already at epoch %d of %d; nothing to train.", start_epoch - 1, cfg.epochs
        )
        start_epoch = cfg.epochs + 1

    # ── training loop ───────────────────────────────────────────────────
    history_path = run_dir / "history.csv"
    write_history_csv(history_path, history)

    best_epoch = 0
    if history["val_macro_f1"]:
        best_epoch = int(history["epoch"][int(np.argmax(history["val_macro_f1"]))])

    training_started = time.time()
    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_started = time.time()
        use_mixup = epoch <= cfg.epochs - cfg.mixup_off_last_epochs
        active_mixup = mixup_fn if use_mixup else None

        train_loss, train_acc = train_one_epoch(
            model,
            ema,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
            active_mixup,
            use_amp,
            cfg.grad_clip,
        )
        raw = evaluate(model, val_loader, criterion, device, use_amp, NUM_CLASSES)
        ema_metrics = evaluate(ema.module, val_loader, criterion, device, use_amp, NUM_CLASSES)

        if ema_metrics["macro_f1"] >= raw["macro_f1"]:
            better, better_source = ema_metrics, "ema"
            better_state = ema.module.state_dict()
        else:
            better, better_source = raw, "raw"
            better_state = model.state_dict()

        epoch_seconds = time.time() - epoch_started
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss_raw"].append(float(raw["loss"]))
        history["val_acc_raw"].append(float(raw["accuracy"]))
        history["val_f1_raw"].append(float(raw["macro_f1"]))
        history["val_loss_ema"].append(float(ema_metrics["loss"]))
        history["val_acc_ema"].append(float(ema_metrics["accuracy"]))
        history["val_f1_ema"].append(float(ema_metrics["macro_f1"]))
        history["val_loss"].append(float(better["loss"]))
        history["val_acc"].append(float(better["accuracy"]))
        history["val_macro_f1"].append(float(better["macro_f1"]))
        history["lr"].append(current_lr)
        history["mixup"].append(int(use_mixup))
        history["epoch_seconds"].append(float(epoch_seconds))
        write_history_csv(history_path, history)

        improved = float(better["macro_f1"]) > best_metric + 1e-6
        if improved:
            best_metric = float(better["macro_f1"])
            best_epoch = epoch
            stale_epochs = 0
            save_weight_checkpoint(
                best_path,
                better_state,
                epoch,
                "best",
                {
                    "val_loss": better["loss"],
                    "val_acc": better["accuracy"],
                    "val_macro_f1": better["macro_f1"],
                    "source": better_source,
                },
                cfg,
            )
        else:
            stale_epochs += 1

        save_weight_checkpoint(
            last_path,
            model.state_dict(),
            epoch,
            "last",
            {
                "val_loss": raw["loss"],
                "val_acc": raw["accuracy"],
                "val_macro_f1": raw["macro_f1"],
                "source": "raw",
            },
            cfg,
        )
        save_resume_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
            ema,
            epoch,
            best_metric,
            stale_epochs,
            history,
            cfg,
        )

        LOGGER.info(
            "Epoch %03d/%03d | train %.4f/%.4f | val raw %.4f/%.4f/%.4f | "
            "ema %.4f/%.4f/%.4f | best %.4f (%s) | %.0fs | mixup=%s",
            epoch,
            cfg.epochs,
            train_loss,
            train_acc,
            raw["loss"],
            raw["accuracy"],
            raw["macro_f1"],
            ema_metrics["loss"],
            ema_metrics["accuracy"],
            ema_metrics["macro_f1"],
            best_metric,
            better_source,
            epoch_seconds,
            "on" if use_mixup else "off",
        )

        if stale_epochs >= cfg.patience:
            LOGGER.info(
                "Early stopping at epoch %d (patience=%d on %s)",
                epoch,
                cfg.patience,
                cfg.monitor,
            )
            break

    LOGGER.info("Training finished in %.1f minutes", (time.time() - training_started) / 60)

    if not best_path.exists():
        LOGGER.error("No best checkpoint was written; aborting evaluation.")
        return 1

    # ── best weights, then infer_size selection on validation ───────────
    best_payload = load_torch(best_path, device)
    model.load_state_dict(best_payload["state_dict"])
    model.eval()
    LOGGER.info(
        "Loaded best checkpoint from epoch %s (val macro-F1 %.4f, %s)",
        best_payload.get("epoch"),
        float(best_payload.get("metrics", {}).get("val_macro_f1", float("nan"))),
        best_payload.get("metrics", {}).get("source"),
    )

    candidate_sizes = sorted({int(cfg.train_size), int(cfg.infer_size_alt)})
    selection: dict[str, dict[str, Any]] = {}
    for size in candidate_sizes:
        for use_tta in (False, True):
            probabilities, y_val = predict_probs(
                model,
                records,
                plan.val,
                device,
                size,
                use_tta,
                cfg.batch_size,
                cfg.num_workers,
                cfg.seed,
                use_amp,
            )
            metrics = compute_metrics(y_val, probabilities, CLASS_NAMES)
            label = f"{size}{'+tta' if use_tta else ''}"
            selection[label] = {
                "size": size,
                "tta": use_tta,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_auc": metrics["macro_auc_ovr"],
            }
            LOGGER.info(
                "  val selection %-8s acc=%.4f f1=%.4f auc=%s",
                label,
                metrics["accuracy"],
                metrics["macro_f1"],
                _fmt(metrics["macro_auc_ovr"]),
            )

    best_label = max(selection, key=lambda key: (selection[key]["macro_f1"], -selection[key]["size"]))
    infer_size = int(selection[best_label]["size"])
    cfg.infer_size = infer_size
    LOGGER.info("Selected infer_size=%d (best validation combo: %s)", infer_size, best_label)

    # ── final test: single scale and 4-view TTA ─────────────────────────
    single_probs, y_test = predict_probs(
        model,
        records,
        plan.test,
        device,
        infer_size,
        False,
        cfg.batch_size,
        cfg.num_workers,
        cfg.seed,
        use_amp,
    )
    single = compute_metrics(y_test, single_probs, CLASS_NAMES)

    tta_sizes = sorted({infer_size, int(cfg.infer_size_alt)})
    tta_probs, y_test_tta = predict_probs(
        model,
        records,
        plan.test,
        device,
        tta_sizes[0],
        True,
        cfg.batch_size,
        cfg.num_workers,
        cfg.seed,
        use_amp,
    )
    if len(tta_sizes) > 1:
        second_probs, _ = predict_probs(
            model,
            records,
            plan.test,
            device,
            tta_sizes[1],
            True,
            cfg.batch_size,
            cfg.num_workers,
            cfg.seed,
            use_amp,
        )
        tta_probs = (tta_probs + second_probs) / 2.0
    tta = compute_metrics(y_test_tta, tta_probs, CLASS_NAMES)

    LOGGER.info(
        "Test single-scale (%d): acc=%.4f f1=%.4f auc=%s",
        infer_size,
        single["accuracy"],
        single["macro_f1"],
        _fmt(single["macro_auc_ovr"]),
    )
    LOGGER.info(
        "Test 4-view TTA %s: acc=%.4f f1=%.4f auc=%s",
        tta_sizes,
        tta["accuracy"],
        tta["macro_f1"],
        _fmt(tta["macro_auc_ovr"]),
    )

    # ── plots ───────────────────────────────────────────────────────────
    plots.plot_training_curves(history, str(plots_dir / "training_curves.png"), best_epoch=best_epoch or None)
    plots.plot_epoch_time(history["epoch_seconds"], str(plots_dir / "epoch_time.png"))
    plots.plot_per_class_f1(tta["per_class"], CLASS_NAMES, str(plots_dir / "per_class_f1.png"))
    plots.plot_per_class_pr(tta["per_class"], CLASS_NAMES, str(plots_dir / "per_class_pr.png"))
    plots.plot_confusion_matrix_counts(tta["confusion"], CLASS_NAMES, str(plots_dir / "confusion_matrix_counts.png"))
    plots.plot_confusion_matrix_normalized(
        tta["confusion"], CLASS_NAMES, str(plots_dir / "confusion_matrix_normalized.png")
    )
    plots.plot_roc_per_class(tta["y_true"], tta_probs, CLASS_NAMES, str(plots_dir / "roc_per_class.png"))
    plots.plot_pr_curves(tta["y_true"], tta_probs, CLASS_NAMES, str(plots_dir / "pr_curves.png"))
    calibration_path, ece = plots.plot_calibration_reliability(
        tta["y_true"], tta_probs, str(plots_dir / "calibration_reliability.png")
    )
    LOGGER.info("Calibration ECE=%.4f -> %s", ece, calibration_path)

    plots.plot_tta_comparison(
        {
            f"single@{infer_size}": {
                "accuracy": single["accuracy"],
                "macro_f1": single["macro_f1"],
                "macro_auc": single["macro_auc_ovr"],
            },
            "tta4": {
                "accuracy": tta["accuracy"],
                "macro_f1": tta["macro_f1"],
                "macro_auc": tta["macro_auc_ovr"],
            },
        },
        str(plots_dir / "tta_comparison.png"),
    )

    # Grad-CAM: one test image per class, predicted class highlighted.
    cam_entries: list[dict[str, Any]] = []
    for class_index, name in enumerate(CLASS_NAMES):
        picks = [index for index in plan.test if records[index].class_index == class_index]
        if not picks:
            continue
        record = records[picks[0]]
        image = np.asarray(load_rgb(record.path))
        tensor = build_eval_transform(infer_size)(load_rgb(record.path)).unsqueeze(0).to(device)
        probabilities, _ = predict_probs(
            model, records, [picks[0]], device, infer_size, False,
            cfg.batch_size, 0, cfg.seed, use_amp,
        )
        predicted = int(np.argmax(probabilities[0]))
        overlay = gradcam_overlay(model, tensor, predicted, image)
        cam_entries.append(
            {
                "class": name,
                "image": image,
                "overlay": overlay,
                "title": f"pred {CLASS_NAMES[predicted]} {probabilities[0][predicted]:.2f}",
            }
        )
    plots.plot_gradcam_grid_per_class(cam_entries, str(plots_dir / "gradcam_grid_per_class.png"))

    # ── sidecar, metrics, packaging ─────────────────────────────────────
    sidecar = {
        "class_names": list(CLASS_NAMES),
        "num_classes": NUM_CLASSES,
        "model_tag": MODEL_TAG,
        "input_size": int(cfg.train_size),
        "infer_size": int(infer_size),
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "metrics": {
            "test_acc": round(float(tta["accuracy"]), 6),
            "test_macro_f1": round(float(tta["macro_f1"]), 6),
            "test_macro_auc_ovr": round(float(tta["macro_auc_ovr"]), 6)
            if tta["macro_auc_ovr"] is not None
            else None,
            "per_class": {
                name: {
                    "precision": round(values["precision"], 6),
                    "recall": round(values["recall"], 6),
                    "f1": round(values["f1"], 6),
                    "support": values["support"],
                }
                for name, values in tta["per_class"].items()
            },
        },
        "thresholds": [0.5, 0.5, 0.5, 0.5],
        "split": {
            "train": len(plan.train),
            "val": len(plan.val),
            "test": len(plan.test),
            "strategy": "phash-cluster-grouped",
        },
    }
    sidecar_path = run_dir / "lung_v2s.label_space.json"
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2)
        handle.write("\n")

    metrics_payload = {
        "model_tag": MODEL_TAG,
        "class_names": list(CLASS_NAMES),
        "seed": cfg.seed,
        "config": cfg.to_dict(),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_metric if best_metric > float("-inf") else None,
        "infer_size": infer_size,
        "val_selection": selection,
        "test_single_scale": {
            "accuracy": single["accuracy"],
            "macro_f1": single["macro_f1"],
            "macro_auc_ovr": single["macro_auc_ovr"],
        },
        "test_tta": {
            "sizes": tta_sizes,
            "views": 2 * len(tta_sizes),
            "accuracy": tta["accuracy"],
            "macro_f1": tta["macro_f1"],
            "macro_auc_ovr": tta["macro_auc_ovr"],
            "per_class": tta["per_class"],
            "confusion": tta["confusion"].tolist(),
            "ece": ece,
        },
        "splits": plan.counts(),
        "split_per_class": plan.per_class,
        "hygiene": {
            "verdict": report["verdict"],
            "exact_duplicate_groups": report["exact_duplicates"]["groups"],
            "near_duplicate_clusters": report["near_duplicates"]["clusters"],
            "cross_class_conflicts": len(report["cross_class_conflicts"]),
            "clusters_spanning_splits": report["leakage"].get("clusters_spanning_splits"),
            "class_counts": report["class_counts"],
            "imbalance_ratio": report["imbalance_ratio"],
        },
        "epochs_run": len(history["epoch"]),
        "history": {key: history[key] for key in HISTORY_KEYS},
    }
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(metrics_payload), handle, indent=2)
        handle.write("\n")

    zip_path = resolve_zip_path(run_dir)
    zipped_path, file_count, byte_size = zip_run_dir(run_dir, zip_path)
    LOGGER.info(
        "Packed %d files into %s (%.2f MB)", file_count, zipped_path, byte_size / (1024 * 1024)
    )

    # ── final summary ───────────────────────────────────────────────────
    per_class_f1 = "  ".join(
        f"{name}={tta['per_class'][name]['f1']:.4f}" for name in CLASS_NAMES
    )
    LOGGER.info("=" * 72)
    LOGGER.info("  RUN COMPLETE — LungV2S")
    LOGGER.info("  best epoch          : %d", best_epoch)
    LOGGER.info("  best val macro-F1   : %.4f", best_metric if best_metric > float("-inf") else 0.0)
    LOGGER.info("  test acc (4-view)   : %.4f", tta["accuracy"])
    LOGGER.info("  test macro-F1       : %.4f", tta["macro_f1"])
    LOGGER.info("  test macro AUC (OvR): %s", _fmt(tta["macro_auc_ovr"]))
    LOGGER.info("  per-class F1        : %s", per_class_f1)
    LOGGER.info("  single-scale acc/F1 : %.4f / %.4f @%dpx", single["accuracy"], single["macro_f1"], infer_size)
    LOGGER.info("  infer_size          : %d (sidecar)", infer_size)
    LOGGER.info("  best checkpoint     : %s", best_path)
    LOGGER.info("  zip                 : %s (%.2f MB, %d files)", zipped_path, byte_size / (1024 * 1024), file_count)
    LOGGER.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
