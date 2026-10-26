"""
TorchXRayVision DenseNet-121 fine-tune — 16-class chest X-ray classification.

Dataset: Kaggle ``h0neyp0t/lungs-class-nih-covid-tb``
    The archive extracts flat: the mount root holds images/, labels.csv, README.md.
    labels.csv columns: image, source, split, <14 NIH findings>, COVID, TB
    split is used as-is (81,797 / 8,612 / 26,027); labels use -1 for "not assessed".

Model:
    xrv.models.DenseNet(weights="densenet121-res224-all")
    classifier -> Dropout(0.3) + Linear(1024, 16)
    op_threshs = None -> raw logits

Input: 1 x 224 x 224, grayscale, scaled to [-1024, 1024] (XRV convention).

Usage:
    python lung_kaggle.py --data_root /kaggle/input --output_dir /kaggle/working

Resume is automatic: rerun the same command after a Kaggle timeout and
training continues from checkpoints/last.pth.  Staged images live in
/tmp/lungs224 and are never written to /kaggle/working.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import csv
import json
import logging
import math
import os
import random
import shutil
import sys
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Optional

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torchxrayvision as xrv

    XRV_VERSION = getattr(xrv, "__version__", "unknown")
except ImportError as exc:  # fresh Kaggle sessions ship without it
    raise RuntimeError(
        "torchxrayvision is required by this trainer but is not installed. "
        "Install it first: `pip install torchxrayvision>=1.2.0` "
        "(on Kaggle run that in a cell, or enable Internet)."
    ) from exc


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lung_xrv")
_T0 = [time.time()]

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CLASSES: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
    "COVID",
    "TB",
)
FINDING_CLASSES = CLASSES[:14]
N_CLASSES = len(CLASSES)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
GROUPS = {
    "findings_14": tuple(range(14)),
    "COVID": (14,),
    "TB": (15,),
}
XRV_WEIGHTS = "densenet121-res224-all"
INPUT_SIZE = 224
XRV_MIN = -1024.0
XRV_MAX = 1024.0
EMA_DECAY = 0.9998
WARMUP_EPOCHS = 2
MIN_LR = 1e-6
BACKBONE_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.02
GRAD_CLIP = 1.0

EXPECTED_SPLIT_COUNTS = {"train": 81_797, "val": 8_612, "test": 26_027}

EXPECTED_POS_WEIGHT = {
    "Atelectasis": 3.08,
    "Cardiomegaly": 7.08,
    "Effusion": 3.01,
    "Infiltration": 2.30,
    "Mass": 4.49,
    "Nodule": 4.18,
    "Pneumonia": 9.89,
    "Pneumothorax": 5.63,
    "Consolidation": 5.41,
    "Edema": 7.82,
    "Emphysema": 7.75,
    "Fibrosis": 8.33,
    "Pleural_Thickening": 6.11,
    "Hernia": 24.43,
    "COVID": 5.22,
    "TB": 11.83,
}

_RESAMPLE = Image.Resampling.LANCZOS


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────


def _quiet_output() -> bool:
    """Kaggle kernels report stderr as a TTY; tqdm there bloats autosave."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return True
    if os.environ.get("OPENMED_NO_PROGRESS"):
        return True
    return not sys.stderr.isatty()


def _progress(iterable, desc: str, total: Optional[int] = None):
    return tqdm(iterable, desc=desc, total=total, leave=False, disable=_quiet_output())


def _save(figure, directory: str, name: str) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def _normalise_rel(name: str) -> PurePosixPath:
    text = name.replace("\\", "/").strip()
    rel = PurePosixPath(text)
    if rel.is_absolute():
        raise ValueError(f"image path must be relative, got {name!r}")
    parts = [part for part in rel.parts if part not in ("", ".")]
    if parts and parts[0].lower() == "images":
        parts = parts[1:]
    if not parts:
        raise ValueError(f"empty image path {name!r}")
    if any(part == ".." for part in parts):
        raise ValueError(f"image path escapes images/: {name!r}")
    return PurePosixPath(*parts)


def _nih_patient_prefix(image_name: str) -> str:
    parts = Path(image_name).stem.split("_")
    return parts[1] if len(parts) > 1 else parts[0]


def _safe_mean(values) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(clean)) if clean else None


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _snapshot(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle mount / labels discovery
# ─────────────────────────────────────────────────────────────────────────────


def _walk_dirs(root: Path, max_depth: int = 6):
    """Breadth-first directory walk; never descends into an images/ folder."""
    root = root.resolve()
    queue: list[tuple[Path, int]] = [(root, 0)]
    seen: set[str] = set()
    while queue:
        current, depth = queue.pop(0)
        key = str(current)
        if key in seen:
            continue
        seen.add(key)
        yield current
        if depth >= max_depth:
            continue
        try:
            entries = sorted(current.iterdir(), key=lambda item: item.name)
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir() and entry.name.lower() != "images":
                queue.append((entry, depth + 1))


def find_labels_csv(data_root: str) -> Path:
    """Walk the input mount, return the single lungs_class/labels.csv."""
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--data_root is not a directory: {root}")

    matches: list[Path] = []
    directories: list[Path] = []
    for directory in _walk_dirs(root):
        directories.append(directory)
        candidate = directory / "labels.csv"
        if candidate.is_file():
            matches.append(candidate)

    if len(matches) != 1:
        listing = "\n  ".join(str(p.relative_to(root)) for p in directories[:40])
        raise RuntimeError(
            f"Expected exactly one labels.csv under {root}, found {len(matches)}: "
            f"{[str(p) for p in matches]}\nScanned directories (first 40):\n  {listing}"
        )
    labels_csv = matches[0]
    logger.info("labels.csv: %s", labels_csv)
    return labels_csv


# ─────────────────────────────────────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Record:
    image: str
    source: str
    split: str
    labels: tuple[float, ...]
    rel: PurePosixPath
    staged_path: str = ""


def read_records(labels_csv: Path) -> tuple[list[Record], Path]:
    images_dir = labels_csv.parent / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images/ beside {labels_csv}")

    records: list[Record] = []
    with open(labels_csv, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        required = ["image", "source", "split", *CLASSES]
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError(f"labels.csv is missing columns: {missing}")

        for line_no, row in enumerate(reader, start=2):
            image = (row.get("image") or "").strip()
            source = (row.get("source") or "").strip() or "unknown"
            split = (row.get("split") or "").strip().lower()
            if split not in {"train", "val", "test"}:
                raise ValueError(f"line {line_no}: invalid split {split!r}")
            if not image:
                raise ValueError(f"line {line_no}: empty image value")

            values: list[float] = []
            for name in CLASSES:
                raw = (row.get(name) or "").strip()
                if raw == "":
                    raise ValueError(f"line {line_no}: missing label {name}")
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"line {line_no}: label {name}={raw!r} is not numeric"
                    ) from exc
                if value not in (-1.0, 0.0, 1.0):
                    raise ValueError(f"line {line_no}: label {name}={value} not in {{-1,0,1}}")
                values.append(value)

            rel = _normalise_rel(image)
            source_path = images_dir / rel
            if not source_path.is_file():
                alt = images_dir / rel.name
                if alt.is_file():
                    rel = PurePosixPath(rel.name)
            records.append(
                Record(
                    image=image,
                    source=source,
                    split=split,
                    labels=tuple(values),
                    rel=rel,
                )
            )

    if not records:
        raise ValueError(f"No rows in {labels_csv}")
    logger.info("Loaded %d label rows from %s", len(records), labels_csv)
    return records, images_dir


def audit_splits(records: list[Record]) -> dict[str, Any]:
    """Fail if an image or a NIH patient crosses splits; log split totals."""
    counts = collections.Counter(record.split for record in records)
    for split in ("train", "val", "test"):
        if counts[split] == 0:
            raise RuntimeError(f"Split {split!r} is empty")

    image_splits: dict[str, set[str]] = collections.defaultdict(set)
    image_rows = collections.Counter()
    for record in records:
        key = record.rel.as_posix()
        image_splits[key].add(record.split)
        image_rows[key] += 1

    leaked_images = {
        key: sorted(splits) for key, splits in image_splits.items() if len(splits) > 1
    }
    if leaked_images:
        examples = list(leaked_images.items())[:10]
        raise RuntimeError(
            f"Leakage: {len(leaked_images)} image(s) occur in more than one split. "
            f"Examples: {examples}"
        )

    patient_splits: dict[str, set[str]] = collections.defaultdict(set)
    patients_by_split: dict[str, set[str]] = {s: set() for s in ("train", "val", "test")}
    for record in records:
        if "nih" not in record.source.lower():
            continue
        prefix = _nih_patient_prefix(record.rel.name)
        patient_splits[prefix].add(record.split)
        patients_by_split[record.split].add(prefix)

    leaked_patients = {
        prefix: sorted(splits) for prefix, splits in patient_splits.items() if len(splits) > 1
    }
    if leaked_patients:
        examples = list(leaked_patients.items())[:10]
        raise RuntimeError(
            f"Leakage: {len(leaked_patients)} NIH patient prefix(es) occur in more than "
            f"one split. Examples: {examples}"
        )

    duplicate_rows = {
        key: count for key, count in image_rows.items() if count > 1
    }
    if duplicate_rows:
        logger.warning(
            "%d image(s) have duplicate rows within a split (not cross-split leakage).",
            len(duplicate_rows),
        )

    split_counts = {split: int(counts[split]) for split in ("train", "val", "test")}
    unique_patients = {
        split: len(patients_by_split[split]) for split in ("train", "val", "test")
    }
    source_counts = collections.Counter(record.source for record in records)

    if split_counts != EXPECTED_SPLIT_COUNTS:
        logger.warning(
            "Split counts %s differ from the expected contract %s; "
            "pos_weight validation will decide whether the dataset is correct.",
            split_counts,
            EXPECTED_SPLIT_COUNTS,
        )

    logger.info(
        "Split audit | train %d / %d patients | val %d / %d patients | "
        "test %d / %d patients",
        split_counts["train"],
        unique_patients["train"],
        split_counts["val"],
        unique_patients["val"],
        split_counts["test"],
        unique_patients["test"],
    )
    logger.info(
        "Leakage audit | cross-split images: 0 | cross-split NIH patient prefixes: 0"
    )

    return {
        "split_counts": split_counts,
        "expected_split_counts": dict(EXPECTED_SPLIT_COUNTS),
        "split_nih_patients": unique_patients,
        "source_counts": {str(k): int(v) for k, v in sorted(source_counts.items())},
        "cross_split_images": 0,
        "cross_split_nih_patients": 0,
        "duplicate_rows_within_split": len(duplicate_rows),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Staging  (/tmp/lungs224, single-channel 224x224 PNG)
# ─────────────────────────────────────────────────────────────────────────────


def stage_images(
    records: list[Record],
    images_dir: Path,
    stage_dir: Path,
    workers: int,
) -> dict[str, int]:
    stage_root = stage_dir.resolve()
    stage_root.mkdir(parents=True, exist_ok=True)

    seen_targets: dict[str, str] = {}
    for record in records:
        target = stage_root / record.rel.with_suffix(".png")
        try:
            target.resolve().relative_to(stage_root)
        except ValueError as exc:
            raise RuntimeError(f"staged path escapes {stage_root}: {target}") from exc
        key = str(target)
        previous = seen_targets.get(key)
        if previous is not None and previous != record.rel.as_posix():
            raise RuntimeError(
                f"Staged filename collision: {previous} and {record.rel.as_posix()} "
                f"both map to {target}"
            )
        seen_targets[key] = record.rel.as_posix()
        record.staged_path = str(target)

    total = len(records)
    already_staged = 0
    for record in records:
        path = Path(record.staged_path)
        if path.is_file() and path.stat().st_size > 0:
            already_staged += 1
    logger.info(
        "Staging: %d/%d already in %s | %d to encode (Kaggle wipes /tmp between "
        "sessions, so a resume may re-stage everything before the first batch)",
        already_staged,
        total,
        stage_root,
        total - already_staged,
    )

    def stage_one(record: Record) -> tuple[str, bool, Optional[str]]:
        target = Path(record.staged_path)
        if target.is_file() and target.stat().st_size > 0:
            return "skipped", False, None

        source = images_dir / record.rel
        if not source.is_file():
            fallback = images_dir / record.rel.name
            if fallback.is_file():
                source = fallback
            else:
                return "error", False, f"missing source for {record.image!r}: {source}"

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        resized = False
        try:
            copy_plain_png = False
            with Image.open(source) as image:
                copy_plain_png = (
                    image.mode == "L"
                    and image.size == (INPUT_SIZE, INPUT_SIZE)
                    and source.suffix.lower() == ".png"
                )
                if not copy_plain_png:
                    converted = image.convert("L")
                    if converted.size != (INPUT_SIZE, INPUT_SIZE):
                        converted = converted.resize(
                            (INPUT_SIZE, INPUT_SIZE), _RESAMPLE
                        )
                        resized = True
                    converted.save(tmp, format="PNG", compress_level=3)
            if copy_plain_png:
                shutil.copyfile(source, tmp)
            os.replace(tmp, target)
        except Exception as exc:  # noqa: BLE001 - reported in aggregate
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            return "error", False, f"{source}: {type(exc).__name__}: {exc}"

        if resized:
            return "resized", True, None
        if copy_plain_png:
            return "copied", False, None
        return "converted", False, None

    summary = {
        "total": len(records),
        "newly_staged": 0,
        "skipped": 0,
        "copied": 0,
        "converted": 0,
        "resized": 0,
    }
    errors: list[str] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        iterator = _progress(
            pool.map(stage_one, records, chunksize=64),
            "[stage]",
            total=len(records),
        )
        for index, (status, resized, error) in enumerate(iterator, start=1):
            if error is not None or status == "error":
                errors.append(error or "unknown staging error")
                continue
            if status == "skipped":
                summary["skipped"] += 1
            else:
                summary["newly_staged"] += 1
                if resized:
                    summary["resized"] += 1
                elif status == "copied":
                    summary["copied"] += 1
                elif status == "converted":
                    summary["converted"] += 1
            if index % 10_000 == 0 or index == total:
                elapsed = time.time() - started
                rate = index / max(elapsed, 1e-6)
                eta_min = (total - index) / max(rate, 1e-6) / 60.0
                logger.info(
                    "  staged %d/%d | %.0f files/s | ETA %.1f min | new %d "
                    "skipped %d resized %d",
                    index,
                    total,
                    rate,
                    eta_min,
                    summary["newly_staged"],
                    summary["skipped"],
                    summary["resized"],
                )

    if errors:
        raise RuntimeError(
            f"Staging failed for {len(errors)} file(s); first errors:\n  "
            + "\n  ".join(errors[:10])
        )

    missing = [r.staged_path for r in records if not Path(r.staged_path).is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} staged files are missing; first: {missing[:5]}")

    logger.info(
        "Staging complete | %d total | %d new (%d copied, %d converted, %d resized) | "
        "%d skipped | %.1f min",
        summary["total"],
        summary["newly_staged"],
        summary["copied"],
        summary["converted"],
        summary["resized"],
        summary["skipped"],
        (time.time() - started) / 60.0,
    )
    if summary["resized"] != 4_316:
        logger.warning(
            "Resized %d images; the contract expects 4,316 COVID/TB files "
            "(NIH should already be 224x224).",
            summary["resized"],
        )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / transforms
# ─────────────────────────────────────────────────────────────────────────────


class ScaleToXRV:
    """[0,1] tensor -> XRV's [-1024, 1024] input range."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.mul(2048.0).sub(1024.0)


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                ScaleToXRV(),
                # value=0.0 = mid-grey: a neutral occlusion inside XRV's range
                transforms.RandomErasing(p=0.2, value=0.0),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            ScaleToXRV(),
        ]
    )


class LungDataset(Dataset):
    def __init__(self, records: list[Record], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.staged_path) as image:
            tensor = self.transform(image.convert("L"))
        target = torch.tensor(record.labels, dtype=torch.float32)
        return tensor, target


# ─────────────────────────────────────────────────────────────────────────────
# Model / loss / EMA
# ─────────────────────────────────────────────────────────────────────────────


def build_model(cache_dir: Path, device: torch.device) -> nn.Module:
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading XRV %s (cache: %s)", XRV_WEIGHTS, cache_dir)
    try:
        model = xrv.models.DenseNet(weights=XRV_WEIGHTS, cache_dir=str(cache_dir))
    except Exception as exc:  # noqa: BLE001 - re-raised with actionable guidance
        raise RuntimeError(
            f"Could not load the XRV '{XRV_WEIGHTS}' weights (a ~28 MB download "
            f"into {cache_dir}). Enable Internet in the Kaggle notebook settings, "
            "or attach the weights as a dataset and point --xrv_cache at its "
            f"folder. Original error: {type(exc).__name__}: {exc}"
        ) from exc
    in_features = int(model.classifier.in_features)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, N_CLASSES))
    model.op_threshs = None  # forward() returns raw logits
    model.apply_sigmoid = False
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model ready | classifier %d -> %d | trainable params %.2fM",
        in_features,
        N_CLASSES,
        trainable / 1e6,
    )
    return model


def compute_pos_weight(records: list[Record]):
    train_rows = [record for record in records if record.split == "train"]
    labels = np.asarray([record.labels for record in train_rows], dtype=np.float64)
    assessed = labels != -1.0
    positives = np.sum((labels == 1.0) & assessed, axis=0)
    negatives = np.sum((labels == 0.0) & assessed, axis=0)
    weights = np.ones(N_CLASSES, dtype=np.float64)
    has_positives = positives > 0
    weights[has_positives] = np.sqrt(
        negatives[has_positives] / positives[has_positives]
    )
    return weights, positives.astype(int), negatives.astype(int), len(train_rows)


def print_and_verify_pos_weight(weights: np.ndarray) -> None:
    logger.info("pos_weight[c] = sqrt(neg_c / pos_c) on TRAIN rows where label != -1")
    for index, name in enumerate(CLASSES):
        logger.info(
            "  %-20s %5.2f   (expected %5.2f)",
            name,
            float(weights[index]),
            EXPECTED_POS_WEIGHT[name],
        )
    expected = np.asarray([EXPECTED_POS_WEIGHT[name] for name in CLASSES], dtype=np.float64)
    delta = np.abs(weights - expected)
    max_delta = float(delta.max())
    if max_delta > 0.05:
        offenders = [
            f"{name}: actual {weights[i]:.3f} vs expected {expected[i]:.3f}"
            for i, name in enumerate(CLASSES)
            if delta[i] > 0.05
        ]
        raise RuntimeError(
            "pos_weight does not match the expected data contract. "
            "Check that the correct labels.csv and train split were loaded.\n  "
            + "\n  ".join(offenders)
        )
    logger.info("pos_weight matches the expected vector (max |delta| = %.4f)", max_delta)


def masked_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked W-BCE. target == -1 contributes no loss and no gradient."""
    mask = targets.ne(-1.0)
    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
    elementwise = F.binary_cross_entropy_with_logits(
        logits, safe_targets, pos_weight=pos_weight, reduction="none"
    )
    masked = elementwise * mask.to(elementwise.dtype)
    mask_count = mask.sum()
    return masked.sum() / mask_count.clamp_min(1), mask_count


def init_ema_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


@torch.no_grad()
def update_ema_state(
    ema_state: dict[str, torch.Tensor], model: nn.Module, decay: float
) -> None:
    for key, value in model.state_dict().items():
        ema = ema_state[key]
        if ema.is_floating_point():
            ema.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        else:
            ema.copy_(value.detach())


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def compute_metrics(probs: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    all_aucs: list[Optional[float]] = []

    for index, name in enumerate(CLASSES):
        mask = targets[:, index] != -1.0
        y = targets[mask, index].astype(np.int64)
        p = probs[mask, index]
        assessed = int(mask.sum())
        positives = int(y.sum())
        negatives = int(len(y) - positives)

        auc: Optional[float] = None
        if len(np.unique(y)) >= 2:
            auc = float(roc_auc_score(y, p))

        prediction = p >= 0.5
        tn = int(np.sum((prediction == 0) & (y == 0)))
        fp = int(np.sum((prediction == 1) & (y == 0)))
        tp = int(np.sum((prediction == 1) & (y == 1)))
        fn = int(np.sum((prediction == 0) & (y == 1)))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else None
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
        accuracy = (tp + tn) / assessed if assessed > 0 else None

        per_class[name] = {
            "auc": auc,
            "assessed": assessed,
            "positives": positives,
            "negatives": negatives,
            "specificity_at_0.5": specificity,
            "sensitivity_at_0.5": sensitivity,
            "accuracy_at_0.5": accuracy,
            "confusion": {"tn": tn, "fp": fp, "tp": tp, "fn": fn},
        }
        all_aucs.append(auc)

    group_means = {
        group: _safe_mean([per_class[CLASSES[i]]["auc"] for i in indices])
        for group, indices in GROUPS.items()
    }
    monitored = _safe_mean(list(group_means.values()))
    mean_auc_all_16 = _safe_mean(all_aucs)

    all_mask = targets != -1.0
    y_all = targets[all_mask].astype(np.int64)
    p_all = probs[all_mask]
    prediction_all = p_all >= 0.5
    tn = int(np.sum((prediction_all == 0) & (y_all == 0)))
    fp = int(np.sum((prediction_all == 1) & (y_all == 0)))
    tp = int(np.sum((prediction_all == 1) & (y_all == 1)))
    fn = int(np.sum((prediction_all == 0) & (y_all == 1)))
    micro_specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    micro_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
    micro_accuracy = (
        (tp + tn) / int(all_mask.sum()) if int(all_mask.sum()) > 0 else None
    )
    macro_specificity = _safe_mean(
        [entry["specificity_at_0.5"] for entry in per_class.values()]
    )
    macro_sensitivity = _safe_mean(
        [entry["sensitivity_at_0.5"] for entry in per_class.values()]
    )

    return {
        "per_class": per_class,
        "group_mean_auc": group_means,
        "monitored_score": monitored if monitored is not None else 0.0,
        "mean_auc_all_16": mean_auc_all_16,
        "specificity_at_0.5": macro_specificity,
        "specificity_at_0.5_macro": macro_specificity,
        "specificity_at_0.5_micro": micro_specificity,
        "sensitivity_at_0.5": macro_sensitivity,
        "sensitivity_at_0.5_micro": micro_sensitivity,
        "accuracy_at_0.5": micro_accuracy,
        "assessed_labels": int(all_mask.sum()),
        "skipped_auc_classes": [
            name for name, entry in per_class.items() if entry["auc"] is None
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────


class GradCAM:
    """Grad-CAM on XRV's final normalisation layer.

    A full-backward hook on ``norm5`` is incompatible with XRV's in-place ReLU
    in ``features2``; capture the activation with a forward hook and compute the
    gradient with ``autograd.grad`` instead.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        target = getattr(model.features, "norm5", model.features)
        self._handles = [target.register_forward_hook(self._save_activation)]

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *exc_info) -> None:
        self.remove_hooks()

    def generate(self, tensor: torch.Tensor, class_index: int) -> np.ndarray:
        if tensor.dim() != 4 or tensor.shape[0] != 1:
            raise ValueError("GradCAM expects a single-image batch")
        was_training = self.model.training
        self.model.eval()
        self.activations = None
        try:
            self.model.zero_grad(set_to_none=True)
            if not tensor.requires_grad:
                tensor = tensor.clone().requires_grad_(True)
            logits = self.model(tensor)
            if self.activations is None:
                raise RuntimeError("Grad-CAM forward hook captured no activation")
            gradients = torch.autograd.grad(
                logits[0, int(class_index)], self.activations
            )[0]
            activations = F.relu(self.activations)
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activations).sum(dim=1)).squeeze(0)
            cam = cam.detach().float().cpu().numpy()
            cam -= float(cam.min())
            peak = float(cam.max())
            if peak > 0:
                cam /= peak
            return cam
        finally:
            self.model.zero_grad(set_to_none=True)
            if was_training:
                self.model.train()


def _denormalise(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().squeeze(0).numpy()
    return np.clip((array + 1024.0) / 2048.0, 0.0, 1.0)


def _overlay_cam(image: np.ndarray, cam: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    height, width = image.shape
    cam_tensor = torch.from_numpy(cam.astype(np.float32))[None, None]
    cam_tensor = F.interpolate(
        cam_tensor, size=(height, width), mode="bilinear", align_corners=False
    )[0, 0]
    cam = cam_tensor.numpy()
    peak = float(cam.max())
    if peak > 0:
        cam = cam / peak
    heat = plt.get_cmap("jet")(np.clip(cam, 0.0, 1.0))[..., :3]
    base = np.repeat(image[..., None], 3, axis=2)
    return np.clip(alpha * heat + (1.0 - alpha) * base, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────


def plot_curves(history: dict[str, list], best_epoch: int, out: str) -> str:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    if len(epochs):
        axes[0].plot(epochs, history["train_loss"], label="train")
        axes[0].plot(epochs, history["val_loss_raw"], label="val raw")
        axes[0].plot(epochs, history["val_loss_ema"], label="val EMA")
        axes[1].plot(epochs, history["val_auc_raw"], label="val raw", color="tab:green")
        axes[1].plot(epochs, history["val_auc_ema"], label="val EMA", color="tab:olive")
        if best_epoch:
            for axis in axes:
                axis.axvline(best_epoch, ls=":", lw=1.0, color="tab:red")
    axes[0].set_title("Masked W-BCE loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].set_title("Monitored AUC | mean(findings, COVID, TB)")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    return _save(figure, out, "training_curves.png")


def plot_per_class_auc(metrics: dict[str, Any], out: str) -> str:
    values = [metrics["per_class"][name]["auc"] for name in CLASSES]
    numeric = [value if value is not None else np.nan for value in values]
    colors = ["tab:blue"] * 14 + ["tab:red", "tab:orange"]
    figure, axis = plt.subplots(figsize=(13, 4.8))
    positions = np.arange(N_CLASSES)
    axis.bar(positions, numeric, color=colors)
    axis.set_xticks(positions)
    axis.set_xticklabels(CLASSES, rotation=35, ha="right", fontsize=8)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("test AUC")
    mean_all = metrics["mean_auc_all_16"]
    axis.axhline(
        mean_all if mean_all is not None else 0.5,
        ls=":",
        color="gray",
        label=f"mean 16: {_fmt(mean_all)}",
    )
    for index, value in enumerate(values):
        if value is not None:
            axis.text(
                index,
                min(value + 0.015, 0.98),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
            )
    groups = metrics["group_mean_auc"]
    axis.set_title(
        "Test AUC per class | findings "
        f"{_fmt(groups['findings_14'])} | COVID {_fmt(groups['COVID'])} | "
        f"TB {_fmt(groups['TB'])}"
    )
    axis.legend(fontsize=8)
    return _save(figure, out, "per_class_auc.png")


def plot_roc_curves(probs: np.ndarray, targets: np.ndarray, out: str) -> str:
    figure, axes = plt.subplots(4, 4, figsize=(16, 16))
    for index, (name, axis) in enumerate(zip(CLASSES, np.ravel(axes), strict=True)):
        mask = targets[:, index] != -1.0
        y = targets[mask, index].astype(np.int64)
        p = probs[mask, index]
        axis.plot([0, 1], [0, 1], "k--", lw=0.7)
        if len(np.unique(y)) >= 2:
            fpr, tpr, _ = roc_curve(y, p)
            auc = roc_auc_score(y, p)
            axis.plot(fpr, tpr, lw=1.2)
            axis.set_title(f"{name}\nAUC {auc:.3f}", fontsize=9)
        else:
            axis.text(
                0.5,
                0.5,
                f"{name}\nskipped\n(n+={int(y.sum())}, n-={int(len(y)-y.sum())})",
                ha="center",
                va="center",
                fontsize=8,
            )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.01)
        axis.tick_params(labelsize=7)
    figure.tight_layout()
    path = os.path.join(out, "roc_curves.png")
    os.makedirs(out, exist_ok=True)
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_gradcam_grid(
    model: nn.Module,
    dataset: LungDataset,
    device: torch.device,
    out: str,
    seed: int = 42,
) -> str:
    rng = random.Random(seed)
    count = min(8, len(dataset))
    picks = rng.sample(range(len(dataset)), count)
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    model.eval()
    with GradCAM(model) as cam_gen:
        for axis, index in zip(np.ravel(axes), picks, strict=False):
            image_tensor, target = dataset[index]
            tensor = image_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                probabilities = torch.sigmoid(model(tensor))[0].float().cpu().numpy()
            top = int(np.argmax(probabilities))
            cam = cam_gen.generate(tensor.clone(), top)
            base = _denormalise(image_tensor)
            overlay = _overlay_cam(base, cam)
            truth = [
                name
                for name, value in zip(CLASSES, target.tolist(), strict=True)
                if float(value) == 1.0
            ]
            assessed = any(float(value) != -1.0 for value in target.tolist())
            truth_text = ", ".join(truth) if truth else ("none" if assessed else "unassessed")
            axis.imshow(overlay)
            axis.axis("off")
            axis.set_title(
                f"pred: {CLASSES[top]} {probabilities[top]:.2f}\ntrue: {truth_text}",
                fontsize=8,
            )
        for axis in np.ravel(axes)[count:]:
            axis.axis("off")
    return _save(figure, out, "gradcam_grid.png")


# ─────────────────────────────────────────────────────────────────────────────
# Optimisation
# ─────────────────────────────────────────────────────────────────────────────


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    backbone = [p for p in model.features.parameters() if p.requires_grad]
    head = [p for p in model.classifier.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": BACKBONE_LR, "name": "backbone"},
            {"params": head, "lr": HEAD_LR, "name": "head"},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer, steps_per_epoch: int, epochs: int
):
    warmup_steps = max(1, WARMUP_EPOCHS * steps_per_epoch)
    total_steps = max(1, steps_per_epoch * epochs)

    def make_lambda(base_lr: float):
        min_ratio = MIN_LR / base_lr

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            denominator = max(1, total_steps - warmup_steps)
            progress = min(max((step - warmup_steps) / denominator, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return lr_lambda

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[make_lambda(BACKBONE_LR), make_lambda(HEAD_LR)],
    )
    return scheduler, total_steps, warmup_steps


def train_one_epoch(
    model: nn.Module,
    ema_state: dict[str, torch.Tensor],
    loader: DataLoader,
    pos_weight: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    limit_batches: int,
) -> float:
    model.train()
    running_loss = 0.0
    running_masked = 0
    max_steps = limit_batches if limit_batches > 0 else len(loader)
    iterator = _progress(loader, f"Epoch {epoch:03d} [train]")
    _T0[0] = time.time()

    for step, (images, targets) in enumerate(iterator, start=1):
        if step > max_steps:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss, mask_count = masked_bce_with_logits(logits, targets, pos_weight)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        update_ema_state(ema_state, model, EMA_DECAY)

        running_loss += float(loss.detach().item()) * int(mask_count.item())
        running_masked += int(mask_count.item())

        if step == 1 or step % 500 == 0 or step == max_steps:
            logger.info(
                "  epoch %03d step %d/%d | loss %.4f | lr backbone %.2e / head %.2e "
                "| %.1f it/s",
                epoch,
                step,
                max_steps,
                float(loss.detach().item()),
                optimizer.param_groups[0]["lr"],
                optimizer.param_groups[1]["lr"],
                step / max(time.time() - _T0[0], 1e-6),
            )

    return running_loss / max(running_masked, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    pos_weight: torch.Tensor,
    device: torch.device,
    tag: str,
    limit_batches: int,
):
    model.eval()
    running_loss = 0.0
    running_masked = 0
    probabilities: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    max_steps = limit_batches if limit_batches > 0 else len(loader)

    for step, (images, targets) in enumerate(_progress(loader, f"[{tag}]"), start=1):
        if step > max_steps:
            break
        images = images.to(device, non_blocking=True)
        targets_device = targets.to(device, non_blocking=True)
        logits = model(images)
        loss, mask_count = masked_bce_with_logits(logits, targets_device, pos_weight)
        running_loss += float(loss.item()) * int(mask_count.item())
        running_masked += int(mask_count.item())
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
        targets_all.append(targets.numpy())

    if not probabilities:
        raise RuntimeError(f"No batches evaluated for {tag}")
    probs = np.concatenate(probabilities, axis=0)
    targets_np = np.concatenate(targets_all, axis=0)
    metrics = compute_metrics(probs, targets_np)
    mean_loss = running_loss / max(running_masked, 1)
    return mean_loss, metrics, probs, targets_np


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoints
# ─────────────────────────────────────────────────────────────────────────────


def _rng_state(device: torch.device, generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
        "shuffle_generator": generator.get_state(),
    }


def save_last_checkpoint(
    path: Path,
    model: nn.Module,
    ema_state: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_score: float,
    best_kind: str,
    best_epoch: int,
    bad_epochs: int,
    history: dict[str, list],
    config: dict[str, Any],
    total_steps: int,
    steps_per_epoch: int,
    device: torch.device,
    generator: torch.Generator,
) -> None:
    torch.save(
        {
            "model": _snapshot(model.state_dict()),
            "ema": _snapshot(ema_state),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "best_kind": best_kind,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "history": copy.deepcopy(history),
            "config": config,
            "classes": list(CLASSES),
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "rng": _rng_state(device, generator),
        },
        path,
    )


def save_best_checkpoint(
    path: Path,
    state: dict[str, torch.Tensor],
    kind: str,
    score: float,
    epoch: int,
    val_loss: float,
    val_metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "state_dict": _snapshot(state),
            "kind": kind,
            "score": score,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "classes": list(CLASSES),
            "config": config,
            "weights": XRV_WEIGHTS,
        },
        path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="TorchXRayVision DenseNet-121 fine-tune on NIH + COVID + TB"
    )
    parser.add_argument("--data_root", default="/kaggle/input")
    parser.add_argument("--output_dir", default="/kaggle/working")
    parser.add_argument(
        "--stage_dir",
        default="/tmp/lungs224",
        help="Single-channel 224x224 staging cache (never under output_dir)",
    )
    parser.add_argument(
        "--xrv_cache",
        default="",
        help="Folder for the XRV weights download (default: <output_dir>/xrv)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--stage_workers", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit_batches",
        type=int,
        default=0,
        help="Smoke-test cap applied to train/val/test loaders; 0 = full split",
    )
    parser.add_argument("--no_resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.limit_batches < 0:
        raise ValueError("--limit_batches must be >= 0")

    output_dir = Path(args.output_dir).expanduser().resolve()
    stage_dir = Path(args.stage_dir).expanduser().resolve()
    if (
        stage_dir == output_dir
        or stage_dir.is_relative_to(output_dir)
        or output_dir.is_relative_to(stage_dir)
    ):
        raise ValueError(
            f"stage_dir ({stage_dir}) and output_dir ({output_dir}) must not overlap; "
            "staged images must never land in /kaggle/working."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    artifacts_dir = output_dir / "artifacts"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    xrv_cache = (
        Path(args.xrv_cache).expanduser().resolve()
        if args.xrv_cache
        else output_dir / "xrv"
    )
    xrv_cache.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required (single GPU only). This trainer does not fall back "
            "to CPU training."
        )
    device = torch.device("cuda:0")
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    logger.info("=" * 72)
    logger.info("  Lung XRV fine-tune | device=%s | seed=%d", torch.cuda.get_device_name(0), args.seed)
    logger.info("  data_root=%s", args.data_root)
    logger.info("  output_dir=%s | stage_dir=%s", output_dir, stage_dir)
    logger.info(
        "  epochs=%d | batch=%d | patience=%d | limit_batches=%d",
        args.epochs,
        args.batch_size,
        args.patience,
        args.limit_batches,
    )
    logger.info("=" * 72)

    labels_csv = find_labels_csv(args.data_root)
    records, images_dir = read_records(labels_csv)
    audit = audit_splits(records)

    pos_weight_np, positives, negatives, train_rows = compute_pos_weight(records)
    print_and_verify_pos_weight(pos_weight_np)
    logger.info(
        "TRAIN rows: %d | positives per class: %s",
        train_rows,
        dict(zip(CLASSES, positives.tolist(), strict=True)),
    )

    staging = stage_images(records, images_dir, stage_dir, args.stage_workers)

    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split == "val"]
    test_records = [record for record in records if record.split == "test"]
    train_dataset = LungDataset(train_records, build_transforms(True))
    val_dataset = LungDataset(val_records, build_transforms(False))
    test_dataset = LungDataset(test_records, build_transforms(False))

    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(args.seed)

    def loader(
        dataset: LungDataset,
        shuffle: bool,
        generator: Optional[torch.Generator] = None,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            persistent_workers=False,
            generator=generator,
        )

    train_loader = loader(train_dataset, True, shuffle_generator)
    val_loader = loader(val_dataset, False)
    test_loader = loader(test_dataset, False)

    steps_per_epoch = (
        min(len(train_loader), args.limit_batches)
        if args.limit_batches > 0
        else len(train_loader)
    )
    if steps_per_epoch < 1:
        raise RuntimeError("Train loader has no batches")

    model = build_model(xrv_cache, device)
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    ema_state = init_ema_state(model)
    optimizer = make_optimizer(model)
    scheduler, total_steps, warmup_steps = make_scheduler(
        optimizer, steps_per_epoch, args.epochs
    )
    scaler = torch.amp.GradScaler("cuda")
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)

    config = {
        "args": vars(args),
        "classes": list(CLASSES),
        "class_groups": {name: list(indices) for name, indices in GROUPS.items()},
        "split_counts": audit["split_counts"],
        "expected_split_counts": audit["expected_split_counts"],
        "split_nih_patients": audit["split_nih_patients"],
        "source_counts": audit["source_counts"],
        "leakage_audit": {
            "cross_split_images": audit["cross_split_images"],
            "cross_split_nih_patients": audit["cross_split_nih_patients"],
            "duplicate_rows_within_split": audit["duplicate_rows_within_split"],
        },
        "pos_weight": {
            "formula": "sqrt(neg_c / pos_c) on TRAIN rows where label != -1",
            "actual": {
                name: float(value)
                for name, value in zip(CLASSES, pos_weight_np, strict=True)
            },
            "expected": dict(EXPECTED_POS_WEIGHT),
            "positives": {name: int(v) for name, v in zip(CLASSES, positives, strict=True)},
            "negatives": {name: int(v) for name, v in zip(CLASSES, negatives, strict=True)},
            "train_rows": train_rows,
        },
        "staging": staging,
        "model": {
            "weights": XRV_WEIGHTS,
            "cache_dir": str(xrv_cache),
            "architecture": "torchxrayvision.models.DenseNet",
            "classifier": "Dropout(0.3) -> Linear(1024, 16)",
            "op_threshs": None,
            "apply_sigmoid": False,
            "input": {
                "shape": [1, INPUT_SIZE, INPUT_SIZE],
                "range": [XRV_MIN, XRV_MAX],
                "scaling": "pixel/255*2048 - 1024",
                "channels": "grayscale",
            },
            "trainable_params": int(
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            ),
        },
        "optimizer": {
            "name": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "param_groups": {
                "backbone": {"lr": BACKBONE_LR},
                "head": {"lr": HEAD_LR},
            },
        },
        "scheduler": {
            "warmup_epochs": WARMUP_EPOCHS,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "cosine_min_lr": MIN_LR,
            "steps_per_epoch": steps_per_epoch,
        },
        "ema": {"decay": EMA_DECAY, "state": "full state_dict including buffers"},
        "loss": (
            "F.binary_cross_entropy_with_logits(logits, safe_target, "
            "pos_weight=pos_weight, reduction='none'); mask=(target != -1); "
            "sum(loss*mask)/sum(mask)"
        ),
        "grad_clip": GRAD_CLIP,
        "amp": "torch.amp.autocast('cuda') + torch.amp.GradScaler('cuda')",
        "augmentation": {
            "train": [
                "Resize(224,224)",
                "RandomHorizontalFlip(0.5)",
                "RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9,1.1))",
                "ColorJitter(brightness=0.2, contrast=0.2)",
                "ToTensor",
                "scale to [-1024,1024]",
                "RandomErasing(0.2)",
            ],
            "eval": ["Resize(224,224)", "ToTensor", "scale to [-1024,1024]"],
        },
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "torchxrayvision": XRV_VERSION,
            "numpy": np.__version__,
            "pillow": __import__("PIL").__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
        "device": str(device),
    }

    last_path = checkpoints_dir / "last.pth"
    best_path = checkpoints_dir / "best.pth"
    log_path = output_dir / "training_log.csv"

    history: dict[str, list] = {
        "train_loss": [],
        "val_loss_raw": [],
        "val_loss_ema": [],
        "val_auc_raw": [],
        "val_auc_ema": [],
        "score_raw": [],
        "score_ema": [],
        "lr": [],
    }
    start_epoch = 1
    best_score = -1.0
    best_kind = ""
    best_epoch = 0
    bad_epochs = 0

    if last_path.is_file() and not args.no_resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if list(checkpoint.get("classes", [])) != list(CLASSES):
            raise RuntimeError("Checkpoint class space does not match this script")
        model.load_state_dict(checkpoint["model"], strict=True)
        ema_state = {
            key: value.to(device) for key, value in checkpoint["ema"].items()
        }
        ema_model.load_state_dict(ema_state, strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", -1.0))
        best_kind = str(checkpoint.get("best_kind", ""))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        bad_epochs = int(checkpoint.get("bad_epochs", 0))
        loaded_history = checkpoint.get("history", {})
        for key in history:
            history[key] = list(loaded_history.get(key, []))
        if int(checkpoint.get("total_steps", total_steps)) != total_steps:
            logger.warning(
                "Resume schedule mismatch: checkpoint total_steps=%s, current=%s. "
                "Use the same --epochs/--batch_size/--limit_batches for an exact resume.",
                checkpoint.get("total_steps"),
                total_steps,
            )
        if int(checkpoint.get("steps_per_epoch", steps_per_epoch)) != steps_per_epoch:
            logger.warning(
                "Resume steps_per_epoch mismatch: checkpoint=%s, current=%s. "
                "An exact resume requires the same loader size.",
                checkpoint.get("steps_per_epoch"),
                steps_per_epoch,
            )
        rng = checkpoint.get("rng")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state(rng["cuda"], device)
            shuffle_generator.set_state(rng["shuffle_generator"])
        logger.info(
            "Resumed %s at epoch %d | best %.4f (%s @ %d) | bad epochs %d",
            last_path,
            start_epoch,
            best_score,
            best_kind or "none",
            best_epoch,
            bad_epochs,
        )
    elif last_path.is_file():
        logger.info("--no_resume set; starting from epoch 1 and overwriting last.pth")

    if not best_path.is_file() and best_score >= 0.0:
        logger.warning("best.pth is missing; clearing best score so epoch 1 re-saves it")
        best_score = -1.0
        best_epoch = 0
        best_kind = ""
        bad_epochs = 0

    write_header = (not log_path.is_file()) or start_epoch == 1
    log_handle = open(  # noqa: SIM115 - kept open across the training loop
        log_path, "w" if write_header else "a", newline=""
    )
    writer = csv.writer(log_handle)
    if write_header:
        writer.writerow(["epoch", "train_loss", "val_loss", "val_mean_auc", "lr", "seconds"])

    training_possible = start_epoch <= args.epochs and bad_epochs < args.patience
    if not training_possible:
        logger.info(
            "No training needed (start_epoch=%d, epochs=%d, bad_epochs=%d, patience=%d)",
            start_epoch,
            args.epochs,
            bad_epochs,
            args.patience,
        )

    if training_possible:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            train_loss = train_one_epoch(
                model,
                ema_state,
                train_loader,
                pos_weight,
                optimizer,
                scheduler,
                scaler,
                device,
                epoch,
                args.limit_batches,
            )

            val_loss_raw, val_metrics_raw, _, _ = evaluate(
                model,
                val_loader,
                pos_weight,
                device,
                "val raw",
                args.limit_batches,
            )
            ema_model.load_state_dict(ema_state, strict=True)
            val_loss_ema, val_metrics_ema, _, _ = evaluate(
                ema_model,
                val_loader,
                pos_weight,
                device,
                "val EMA",
                args.limit_batches,
            )

            score_raw = float(val_metrics_raw["monitored_score"])
            score_ema = float(val_metrics_ema["monitored_score"])
            if score_ema > score_raw:
                candidate_score = score_ema
                candidate_kind = "ema"
                candidate_state = ema_state
                candidate_metrics = val_metrics_ema
                candidate_loss = val_loss_ema
            else:
                candidate_score = score_raw
                candidate_kind = "raw"
                candidate_state = model.state_dict()
                candidate_metrics = val_metrics_raw
                candidate_loss = val_loss_raw

            improved = best_epoch == 0 or candidate_score > best_score
            if improved:
                best_score = float(candidate_score)
                best_kind = candidate_kind
                best_epoch = epoch
                bad_epochs = 0
                save_best_checkpoint(
                    best_path,
                    candidate_state,
                    candidate_kind,
                    candidate_score,
                    epoch,
                    candidate_loss,
                    candidate_metrics,
                    config,
                )
            else:
                bad_epochs += 1

            history["train_loss"].append(float(train_loss))
            history["val_loss_raw"].append(float(val_loss_raw))
            history["val_loss_ema"].append(float(val_loss_ema))
            history["val_auc_raw"].append(score_raw)
            history["val_auc_ema"].append(score_ema)
            history["score_raw"].append(score_raw)
            history["score_ema"].append(score_ema)
            history["lr"].append(float(optimizer.param_groups[1]["lr"]))

            save_last_checkpoint(
                last_path,
                model,
                ema_state,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_score,
                best_kind,
                best_epoch,
                bad_epochs,
                history,
                config,
                total_steps,
                steps_per_epoch,
                device,
                shuffle_generator,
            )

            elapsed = time.time() - epoch_start
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.4f}",
                    f"{val_loss_raw:.4f}",
                    f"{score_raw:.4f}",
                    f"{optimizer.param_groups[1]['lr']:.6f}",
                    f"{elapsed:.1f}",
                ]
            )
            log_handle.flush()
            logger.info(
                "Epoch %03d/%03d | train %.4f | val raw %.4f AUC %.4f | "
                "EMA %.4f AUC %.4f | groups raw [findings %s COVID %s TB %s] | "
                "monitored %.4f (%s) | lr %.2e | best %.4f @%d | %.0fs",
                epoch,
                args.epochs,
                train_loss,
                val_loss_raw,
                score_raw,
                val_loss_ema,
                score_ema,
                _fmt(val_metrics_raw["group_mean_auc"]["findings_14"]),
                _fmt(val_metrics_raw["group_mean_auc"]["COVID"]),
                _fmt(val_metrics_raw["group_mean_auc"]["TB"]),
                candidate_score,
                candidate_kind,
                best_score,
                best_epoch,
                elapsed,
            )

            if bad_epochs >= args.patience:
                logger.info(
                    "Early stopping at epoch %d (patience %d on monitored AUC).",
                    epoch,
                    args.patience,
                )
                break

    log_handle.close()

    if not best_path.is_file():
        raise RuntimeError("No best checkpoint exists; cannot evaluate the test split")

    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["state_dict"], strict=True)
    model.eval()
    test_loss, test_metrics, test_probs, test_targets = evaluate(
        model,
        test_loader,
        pos_weight,
        device,
        "test",
        args.limit_batches,
    )

    logger.info(
        "TEST | best epoch %d (%s) | loss %.4f | mean AUC 16 %s | "
        "findings %s | COVID %s | TB %s | monitored %s",
        int(best_checkpoint["epoch"]),
        best_checkpoint["kind"],
        test_loss,
        _fmt(test_metrics["mean_auc_all_16"]),
        _fmt(test_metrics["group_mean_auc"]["findings_14"]),
        _fmt(test_metrics["group_mean_auc"]["COVID"]),
        _fmt(test_metrics["group_mean_auc"]["TB"]),
        _fmt(test_metrics["monitored_score"]),
    )
    logger.info(
        "TEST specificity@0.5 | macro %s | micro %s | sensitivity macro %s | "
        "accuracy %s",
        _fmt(test_metrics["specificity_at_0.5_macro"]),
        _fmt(test_metrics["specificity_at_0.5_micro"]),
        _fmt(test_metrics["sensitivity_at_0.5"]),
        _fmt(test_metrics["accuracy_at_0.5"]),
    )
    logger.info("TEST per-class AUC / sensitivity@0.5 / specificity@0.5 / support:")
    for name in CLASSES:
        entry = test_metrics["per_class"][name]
        logger.info(
            "  %-20s AUC %s | sens %s | spec %s | n+=%d n-=%d",
            name,
            _fmt(entry["auc"]),
            _fmt(entry["sensitivity_at_0.5"]),
            _fmt(entry["specificity_at_0.5"]),
            entry["positives"],
            entry["negatives"],
        )
    if test_metrics["skipped_auc_classes"]:
        logger.warning(
            "Test AUC skipped (fewer than 2 distinct labels): %s",
            test_metrics["skipped_auc_classes"],
        )

    artifact_paths = [
        plot_curves(history, best_epoch, str(artifacts_dir)),
        plot_per_class_auc(test_metrics, str(artifacts_dir)),
        plot_roc_curves(test_probs, test_targets, str(artifacts_dir)),
        plot_gradcam_grid(model, test_dataset, device, str(artifacts_dir), args.seed),
    ]

    metrics = {
        "model": config["model"],
        "weights": XRV_WEIGHTS,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_run": len(history["train_loss"]),
        "limit_batches": args.limit_batches,
        "best": {
            "epoch": int(best_checkpoint["epoch"]),
            "kind": best_checkpoint["kind"],
            "monitored_score": float(best_checkpoint["score"]),
            "val_loss": float(best_checkpoint["val_loss"]),
            "val_metrics": best_checkpoint["val_metrics"],
        },
        "test_loss": float(test_loss),
        "test": test_metrics,
        "split_counts": audit["split_counts"],
        "split_audit": audit,
        "pos_weight": config["pos_weight"],
        "staging": staging,
        "config": config,
        "history": history,
        "artifacts": [
            "training_log.csv",
            "artifacts/training_curves.png",
            "artifacts/per_class_auc.png",
            "artifacts/roc_curves.png",
            "artifacts/gradcam_grid.png",
            "artifacts/metrics.json",
            "checkpoints/best.pth",
            "checkpoints/last.pth",
            "lung_xrv_results.zip",
        ],
        "artifact_files_before_zip": artifact_paths,
    }
    metrics_path = artifacts_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(metrics), handle, indent=2)

    zip_path = output_dir / "lung_xrv_results.zip"
    entries = [
        (str(log_path), "training_log.csv"),
        (str(best_path), "checkpoints/best.pth"),
        (str(last_path), "checkpoints/last.pth"),
        (str(artifacts_dir / "training_curves.png"), "artifacts/training_curves.png"),
        (str(artifacts_dir / "per_class_auc.png"), "artifacts/per_class_auc.png"),
        (str(artifacts_dir / "roc_curves.png"), "artifacts/roc_curves.png"),
        (str(artifacts_dir / "gradcam_grid.png"), "artifacts/gradcam_grid.png"),
        (str(metrics_path), "artifacts/metrics.json"),
    ]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, arcname in entries:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing artifact for zip: {path}")
            archive.write(path, arcname)

    zip_size = zip_path.stat().st_size
    if zip_size > 1024**3:
        raise RuntimeError(f"Artifact zip exceeds 1 GB: {zip_size / 1024**3:.2f} GB")

    logger.info("=" * 72)
    logger.info("  Run complete")
    logger.info("  best       : %s (epoch %d, %s)", best_path, best_epoch, best_kind)
    logger.info("  metrics    : %s", metrics_path)
    logger.info("  zip        : %s (%.1f MB)", zip_path, zip_size / 1024**2)
    logger.info(
        "  test AUC   : mean16 %s | findings %s | COVID %s | TB %s | spec@0.5 %s",
        _fmt(test_metrics["mean_auc_all_16"]),
        _fmt(test_metrics["group_mean_auc"]["findings_14"]),
        _fmt(test_metrics["group_mean_auc"]["COVID"]),
        _fmt(test_metrics["group_mean_auc"]["TB"]),
        _fmt(test_metrics["specificity_at_0.5"]),
    )
    logger.info("  artifacts  : %s", ", ".join(metrics["artifacts"]))
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
