"""Fit a temperature-scaling scalar on the validation split.

Why this exists
---------------
The 4-class LungV2S head is trained with class-weighted cross-entropy, mixup
and label smoothing. All three deliberately distort the softmax: the weighted
loss shifts the implied class priors, mixup makes the model hedge, and label
smoothing softens targets. The result is a head that classifies well (test
macro-F1 ~0.96) but whose *confidence* is not a probability — the first run
measured ECE 0.2014, i.e. "90% sure" was really ~70% correct.

Temperature scaling fixes exactly that and nothing else. A single scalar T,
fitted by minimising NLL on held-out validation logits, rescales the softmax:

    p = softmax(logits / T)

Because the division is monotonic, the argmax (and therefore accuracy, F1 and
every confusion-matrix cell) is unchanged. Only the confidence number moves,
and it moves toward the truth. T is written into the label-space sidecar, which
``pipeline.py`` reads and applies before softmax at serving time.

This script is data-only: it needs the checkpoint, the sidecar, the split
manifest and the raw images. It does not touch the trainer and does not
retrain. Run it on Kaggle after a training run, or locally against a downloaded
run directory.

Usage (from ``backend/``)::

    venv/bin/python -m app.organs.lungs.classification.kaggle.fit_temperature \
        --checkpoint app/organs/lungs/classification/checkpoints/lung_v2s_best.pth \
        --manifest   app/organs/lungs/classification/checkpoints/training_run/split_manifest.csv \
        --data-root  ~/Downloads/COVID-19_Radiography_Dataset \
        --split val
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from PIL import Image

try:  # package import: app.organs.lungs.classification.kaggle.fit_temperature
    from ..model_v2s import build_model, load_sidecar, preprocess_v2s
except ImportError:  # direct execution
    _ROOT = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(_ROOT / "backend"))
    from app.organs.lungs.classification.model_v2s import (  # type: ignore
        build_model,
        load_sidecar,
        preprocess_v2s,
    )

LOGGER = logging.getLogger("lung_v2s.fit_temperature")


# ── data ─────────────────────────────────────────────────────────────────────


def _read_manifest(manifest_path: Path, split: str, class_names: Sequence[str]) -> list[tuple[str, int]]:
    """Return ``(relative_path, label_index)`` for every row in ``split``."""
    index_of = {name: i for i, name in enumerate(class_names)}
    rows: list[tuple[str, int]] = []
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "class", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"manifest {manifest_path} is missing columns {sorted(missing)}"
            )
        for row in reader:
            if row["split"] != split:
                continue
            label = index_of.get(row["class"])
            if label is None:
                raise ValueError(
                    f"manifest class {row['class']!r} is not in the sidecar class_names "
                    f"{list(class_names)}"
                )
            rows.append((row["path"], label))
    if not rows:
        raise ValueError(f"no rows for split {split!r} in {manifest_path}")
    return rows


@torch.no_grad()
def _collect_logits(
    model: torch.nn.Module,
    rows: Sequence[tuple[str, int]],
    data_root: Path,
    infer_size: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over ``rows`` and return ``(logits, labels)``."""
    logits_out: list[np.ndarray] = []
    labels_out: list[int] = []
    total = len(rows)
    started = time.time()
    batch_tensors: list[torch.Tensor] = []
    batch_labels: list[int] = []

    def flush() -> None:
        if not batch_tensors:
            return
        batch = torch.cat(batch_tensors, dim=0).to(device)
        logits = model(batch)
        logits_out.append(logits.detach().float().cpu().numpy())
        labels_out.extend(batch_labels)
        batch_tensors.clear()
        batch_labels.clear()

    for position, (rel_path, label) in enumerate(rows, start=1):
        image_path = data_root / rel_path
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        batch_tensors.append(preprocess_v2s(image, infer_size))
        batch_labels.append(label)
        if len(batch_tensors) >= batch_size:
            flush()
        if position % 500 == 0 or position == total:
            elapsed = time.time() - started
            rate = position / elapsed if elapsed > 0 else 0.0
            LOGGER.info("  scored %d/%d (%.1f img/s)", position, total, rate)
    flush()

    return np.concatenate(logits_out, axis=0), np.asarray(labels_out, dtype=np.int64)


# ── calibration maths ────────────────────────────────────────────────────────


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Standard top-label ECE over equal-width confidence bins."""
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correctness = (prediction == labels).astype(np.float64)
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        if high == 1.0:
            in_bin = (confidence >= low) & (confidence <= high)
        else:
            in_bin = (confidence >= low) & (confidence < high)
        count = int(in_bin.sum())
        if count == 0:
            continue
        acc = float(correctness[in_bin].mean())
        conf = float(confidence[in_bin].mean())
        ece += abs(acc - conf) * count / len(labels)
    return float(ece)


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, max_iter: int = 200
) -> float:
    """Return the scalar T that minimises NLL of ``softmax(logits / T)``.

    LBFGS on a single positive parameter. ``log_T`` is optimised so T stays
    positive without clamping artefacts.
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.05, max_iter=max_iter, line_search_fn="strong_wolfe"
    )
    criterion = torch.nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min=1e-3, max=1e3)
        loss = criterion(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).clamp(min=1e-3, max=1e3).item())


# ── sidecar update ───────────────────────────────────────────────────────────


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                break
        else:
            state = payload
    else:
        raise ValueError("checkpoint does not contain a state dict")
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


# ── entrypoint ───────────────────────────────────────────────────────────────


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit temperature scaling for LungV2S")
    parser.add_argument("--checkpoint", required=True, help="lung_v2s_best.pth")
    parser.add_argument(
        "--sidecar",
        default=None,
        help="label-space json; defaults to the sidecar beside the checkpoint",
    )
    parser.add_argument("--manifest", required=True, help="split_manifest.csv")
    parser.add_argument("--data-root", required=True, help="root holding <class>/images/*.png")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--limit", type=int, default=0, help="score only the first N rows (0 = all)"
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="optional path for a calibrated reliability PNG",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    sidecar_path = (
        Path(args.sidecar).expanduser().resolve()
        if args.sidecar
        else checkpoint_path.with_name("lung_v2s.label_space.json")
    )
    manifest_path = Path(args.manifest).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()

    sidecar = load_sidecar(str(sidecar_path))
    class_names = list(sidecar["class_names"])
    infer_size = int(sidecar["infer_size"])
    LOGGER.info(
        "Sidecar: %d classes %s, infer_size=%d", len(class_names), class_names, infer_size
    )

    device = torch.device(args.device)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

    rows = _read_manifest(manifest_path, args.split, class_names)
    if args.limit and args.limit < len(rows):
        rows = rows[: args.limit]
    LOGGER.info("Scoring %d %s images from %s", len(rows), args.split, data_root)

    model = build_model(num_classes=len(class_names), pretrained=False)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(_extract_state_dict(payload), strict=True)
    model.eval().to(device)

    logits, labels = _collect_logits(
        model, rows, data_root, infer_size, device, args.batch_size
    )

    ece_before = expected_calibration_error(_softmax(logits, 1.0), labels)
    temperature = fit_temperature(logits, labels)
    ece_after = expected_calibration_error(_softmax(logits, temperature), labels)

    LOGGER.info(
        "Fitted temperature T=%.4f on %d %s images | ECE %.4f -> %.4f",
        temperature,
        len(labels),
        args.split,
        ece_before,
        ece_after,
    )

    sidecar["temperature"] = round(float(temperature), 6)
    calibration = dict(sidecar.get("calibration") or {})
    calibration.update(
        {
            "method": "temperature-scaling",
            "fit_split": args.split,
            "n": int(len(labels)),
            "ece_before": round(float(ece_before), 6),
            "ece_after": round(float(ece_after), 6),
            "temperature": round(float(temperature), 6),
        }
    )
    sidecar["calibration"] = calibration
    _write_json_atomic(sidecar_path, sidecar)
    LOGGER.info("Wrote temperature into %s", sidecar_path)

    if args.plot:
        try:
            from . import plots  # local import: matplotlib only when plotting

            plots.plot_calibration_reliability(
                labels, _softmax(logits, temperature), args.plot
            )
            LOGGER.info("Wrote calibrated reliability plot -> %s", args.plot)
        except Exception as exc:  # pragma: no cover - plotting is best-effort
            LOGGER.warning("Could not write reliability plot: %s", exc)

    print(
        json.dumps(
            {
                "temperature": sidecar["temperature"],
                "ece_before": calibration["ece_before"],
                "ece_after": calibration["ece_after"],
                "split": args.split,
                "n": calibration["n"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
