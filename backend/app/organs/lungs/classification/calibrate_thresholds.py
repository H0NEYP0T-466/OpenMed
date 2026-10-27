"""Calibrate per-class decision thresholds on the validation split.

The trained head outputs posteriors that are not centred on 0.5: class
weighting shifts the distribution, so a global 0.5 threshold yields ~0.46
sensitivity at 0.93 specificity. This script picks a threshold per class from
the validation split and writes ``thresholds.json`` beside the checkpoint,
which the serving pipeline then applies.

Two operating points are computed for every class:

* ``target_sensitivity`` (default 0.80) — the highest threshold that still
  catches 80 % of positives. Use this for screening.
* Youden's J — the threshold maximising (sensitivity + specificity − 1).

The file records both; serving uses ``target_sensitivity`` unless configured
otherwise.

Kaggle usage::

    !python calibrate_thresholds.py --data_root /tmp/data \
        --checkpoint /kaggle/working/checkpoints/best.pth \
        --output /kaggle/working/checkpoints/thresholds.json

Nothing here trains anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_curve

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lung_thresholds")

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
INPUT_SIZE = 224
XRV_WEIGHTS = "densenet121-res224-all"


def find_labels_csv(root: str) -> Path:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"--data_root is not a directory: {root_path}")
    for directory, _dirs, files in os.walk(root_path):
        if "labels.csv" in files:
            return Path(directory) / "labels.csv"
    raise FileNotFoundError(f"No labels.csv under {root_path}")


def read_val_rows(labels_csv: Path) -> list[tuple[str, np.ndarray]]:
    rows: list[tuple[str, np.ndarray]] = []
    with open(labels_csv, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("split") or "").strip().lower() != "val":
                continue
            labels = np.array([float(row[name]) for name in CLASSES], dtype=np.float32)
            rows.append((row["image"], labels))
    return rows


def build_model(cache_dir: str) -> nn.Module:
    import torchxrayvision as xrv

    model = xrv.models.DenseNet(weights=XRV_WEIGHTS, cache_dir=cache_dir)
    in_features = int(model.classifier.in_features)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, len(CLASSES)))
    model.op_threshs = None
    model.apply_sigmoid = False
    return model


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        gray = image.convert("L").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    array = np.asarray(gray, dtype=np.float32) / 255.0
    return array * 2048.0 - 1024.0


def predict_all(
    model: nn.Module,
    rows: list[tuple[str, np.ndarray]],
    images_dir: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    def load(item: tuple[str, np.ndarray]) -> np.ndarray:
        return load_image(images_dir / item[0])

    probabilities: list[np.ndarray] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            arrays = list(pool.map(load, chunk))
            tensor = torch.from_numpy(np.stack(arrays)).unsqueeze(1).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
            done = min(start + batch_size, len(rows))
            if done % 1000 == 0 or done == len(rows):
                rate = done / max(time.perf_counter() - started, 1e-6)
                logger.info("  scored %d/%d | %.1f img/s", done, len(rows), rate)
    return np.concatenate(probabilities, axis=0)


def calibrate(
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_sensitivity: float,
) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    for index, name in enumerate(CLASSES):
        mask = targets[:, index] != -1.0
        y = targets[mask, index].astype(int)
        p = probabilities[mask, index]
        positives = int(y.sum())
        negatives = int(len(y) - positives)
        if positives < 2 or negatives < 2:
            per_class[name] = {
                "threshold": 0.5,
                "youden_threshold": None,
                "sensitivity": None,
                "specificity": None,
                "positives": positives,
                "negatives": negatives,
                "note": "insufficient support — kept at 0.5",
            }
            continue

        fpr, tpr, thresholds = roc_curve(y, p)
        # roc_curve prepends an implicit infinity threshold; drop it.
        valid = np.isfinite(thresholds)
        fpr, tpr, thresholds = fpr[valid], tpr[valid], thresholds[valid]

        youden_index = int(np.argmax(tpr - fpr))
        youden_threshold = float(thresholds[youden_index])

        # Highest threshold that still reaches the target sensitivity: the
        # tightest operating point that does not trade away recall.
        meeting = np.where(tpr >= target_sensitivity)[0]
        if meeting.size:
            pick = int(meeting[np.argmax(thresholds[meeting])])
        else:
            pick = youden_index
        threshold = float(thresholds[pick])

        def rates(cut: float) -> tuple[float, float]:
            predicted = p >= cut
            tp = int(((predicted == 1) & (y == 1)).sum())
            tn = int(((predicted == 0) & (y == 0)).sum())
            return tp / max(positives, 1), tn / max(negatives, 1)

        sensitivity, specificity = rates(threshold)
        per_class[name] = {
            "threshold": round(threshold, 4),
            "youden_threshold": round(youden_threshold, 4),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "positives": positives,
            "negatives": negatives,
        }
    return per_class


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-class threshold calibration")
    parser.add_argument("--data_root", default="/kaggle/input")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--target_sensitivity", type=float, default=0.80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.warning("No GPU — CPU scoring of 8,612 images will take hours.")

    labels_csv = find_labels_csv(args.data_root)
    images_dir = labels_csv.parent / "images"
    rows = read_val_rows(labels_csv)
    if not rows:
        raise RuntimeError("No validation rows found in labels.csv")
    logger.info("Validation rows: %d | images: %s", len(rows), images_dir)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else checkpoint.with_name("thresholds.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(str(checkpoint.parent / "xrv"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    logger.info(
        "Loaded %s (epoch %s, %s)",
        checkpoint.name,
        payload.get("epoch") if isinstance(payload, dict) else None,
        payload.get("kind") if isinstance(payload, dict) else None,
    )

    probabilities = predict_all(model, rows, images_dir, device, args.batch_size, args.workers)
    targets = np.stack([labels for _, labels in rows])
    per_class = calibrate(probabilities, targets, args.target_sensitivity)

    payload_out = {
        "classes": list(CLASSES),
        "target_sensitivity": args.target_sensitivity,
        "checkpoint": checkpoint.name,
        "split": "val",
        "rows": len(rows),
        "per_class": per_class,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload_out, indent=2))

    logger.info("=" * 66)
    logger.info("  %-20s %8s %8s %8s", "class", "thresh", "sens", "spec")
    for name in CLASSES:
        entry = per_class[name]
        logger.info(
            "  %-20s %8.3f %8s %8s",
            name,
            entry["threshold"],
            f"{entry['sensitivity']:.3f}" if entry["sensitivity"] is not None else "n/a",
            f"{entry['specificity']:.3f}" if entry["specificity"] is not None else "n/a",
        )
    logger.info("  written: %s", output)
    logger.info("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
