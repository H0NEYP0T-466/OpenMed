"""Zero-shot baseline: pretrained ``densenet121-res224-all`` on the same split.

The fine-tuned model must be compared against the weights it started from, not
against a number from a paper. This script runs the *untouched* TorchXRayVision
checkpoint over the test split and reports AUC on the fourteen NIH classes —
the only classes that checkpoint knows.

Its 18 outputs are a superset of our 14, so the comparison is a subset, not a
remap: no class is renamed, no output is reinterpreted.

Kaggle usage::

    !python zero_shot_baseline.py --data_root /tmp/data \
        --output /kaggle/working/artifacts/zero_shot.json

Nothing here trains or fine-tunes anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("zero_shot")

NIH_FINDINGS: tuple[str, ...] = (
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
)
INPUT_SIZE = 224
XRV_WEIGHTS = "densenet121-res224-all"


def find_labels_csv(root: str) -> Path:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"--data_root is not a directory: {root_path}")
    found = [Path(d) / "labels.csv" for d, _s, files in os.walk(root_path) if "labels.csv" in files]
    if len(found) != 1:
        raise RuntimeError(f"Expected exactly one labels.csv under {root_path}, found {len(found)}")
    return found[0]


def load_split(labels_csv: Path, split: str, limit: int) -> list[tuple[str, np.ndarray]]:
    rows: list[tuple[str, np.ndarray]] = []
    with open(labels_csv, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("split") or "").strip().lower() != split:
                continue
            labels = np.array([float(row[name]) for name in NIH_FINDINGS], dtype=np.float32)
            rows.append((row["image"], labels))
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Zero-shot XRV baseline on the test split")
    parser.add_argument("--data_root", default="/kaggle/input")
    parser.add_argument("--output", default="zero_shot.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Cap rows for a smoke run; 0 = all")
    parser.add_argument("--cache_dir", default="")
    args = parser.parse_args()

    import torchxrayvision as xrv

    labels_csv = find_labels_csv(args.data_root)
    images_dir = labels_csv.parent / "images"
    rows = load_split(labels_csv, args.split, args.limit)
    if not rows:
        raise RuntimeError(f"No rows for split {args.split!r}")
    logger.info("%s rows: %d | images: %s", args.split, len(rows), images_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = args.cache_dir or str(Path(args.output).parent / "xrv")
    model = xrv.models.DenseNet(weights=XRV_WEIGHTS, cache_dir=cache_dir)
    model.eval().to(device)
    labels_out = list(model.pathologies)
    index = {name: i for i, name in enumerate(labels_out)}
    missing = [name for name in NIH_FINDINGS if name not in index]
    if missing:
        raise RuntimeError(f"Checkpoint lacks these classes: {missing}")
    logger.info("checkpoint classes: %d | using %d of them", len(labels_out), len(NIH_FINDINGS))

    def load(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            gray = image.convert("L").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        array = np.asarray(gray, dtype=np.float32) / 255.0
        return array * 2048.0 - 1024.0

    probabilities: list[np.ndarray] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start : start + args.batch_size]
            arrays = list(pool.map(lambda r: load(images_dir / r[0]), chunk))
            tensor = torch.from_numpy(np.stack(arrays)).unsqueeze(1).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
            done = min(start + args.batch_size, len(rows))
            if done % 2000 == 0 or done == len(rows):
                logger.info("  scored %d/%d | %.1f img/s", done, len(rows), done / max(time.perf_counter() - started, 1e-6))

    probs = np.concatenate(probabilities, axis=0)
    targets = np.stack([labels for _, labels in rows])

    per_class: dict[str, Any] = {}
    for column, name in enumerate(NIH_FINDINGS):
        mask = targets[:, column] != -1.0
        y = targets[mask, column].astype(int)
        p = probs[mask, index[name]]
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) >= 2 else None
        per_class[name] = {"auc": auc, "positives": int(y.sum()), "negatives": int(len(y) - y.sum())}

    aucs = [v["auc"] for v in per_class.values() if v["auc"] is not None]
    mean_auc = float(np.mean(aucs)) if aucs else None

    payload = {
        "model": XRV_WEIGHTS,
        "mode": "zero-shot",
        "split": args.split,
        "rows": len(rows),
        "mean_auc_14": mean_auc,
        "per_class": per_class,
        "note": (
            "Pretrained weights, no fine-tuning. Classes COVID and TB do not "
            "exist in this checkpoint and are excluded."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))

    logger.info("=" * 60)
    for name, entry in per_class.items():
        logger.info("  %-20s AUC %s (n+=%d n-=%d)", name,
                    f"{entry['auc']:.4f}" if entry["auc"] is not None else "n/a",
                    entry["positives"], entry["negatives"])
    logger.info("  mean AUC over 14: %.4f", mean_auc if mean_auc else float("nan"))
    logger.info("  written: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
