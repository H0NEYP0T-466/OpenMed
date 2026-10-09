"""Segmentation metrics.

Two tiers:
  * ``batch_stats`` — cheap per-image TP/FP/FN counts on the accelerator, used
    every epoch for validation (Dice, IoU, precision, recall);
  * ``hd95`` — Hausdorff-95 on the CPU, used only for the final test.

Conventions: HD95 is in *pixels* (no validated pixel spacing exists for these
slices) and is ``nan`` when exactly one of prediction and truth is empty —
substituting a large finite number would look like a measurement.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy import ndimage

SIZE_BINS = (
    ("tiny  (<0.5%)", 0.0, 0.005),
    ("small (0.5-1.5%)", 0.005, 0.015),
    ("medium (1.5-4%)", 0.015, 0.04),
    ("large (>4%)", 0.04, 1.01),
)


@torch.no_grad()
def batch_stats(probs: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Per-image ``[tp, fp, fn, pred_px, gt_px]`` as float64, shape ``(B, 5)``."""
    pred = probs >= threshold
    gt = target >= 0.5
    dims = tuple(range(1, pred.ndim))
    tp = (pred & gt).sum(dims)
    fp = (pred & ~gt).sum(dims)
    fn = (~pred & gt).sum(dims)
    return torch.stack([tp, fp, fn, pred.sum(dims), gt.sum(dims)], dim=1).double()


def scores_from_stats(stats: np.ndarray) -> dict[str, np.ndarray]:
    """Dice, IoU, precision, recall per image from an ``(N, 5)`` stats array."""
    tp, fp, fn, pred_px, gt_px = (stats[:, i] for i in range(5))
    eps = 1e-9
    both_empty = (pred_px == 0) & (gt_px == 0)
    dice = np.where(both_empty, 1.0, 2 * tp / np.maximum(2 * tp + fp + fn, eps))
    iou = np.where(both_empty, 1.0, tp / np.maximum(tp + fp + fn, eps))
    precision = np.where(pred_px > 0, tp / np.maximum(tp + fp, eps), np.where(gt_px == 0, 1.0, 0.0))
    recall = np.where(gt_px > 0, tp / np.maximum(tp + fn, eps), 1.0)
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


def summarise(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "p5": None, "p95": None}
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": int(values.size), "mean": None, "median": None, "std": None, "p5": None, "p95": None}
    return {
        "n": int(values.size), "n_finite": int(finite.size),
        "mean": float(finite.mean()), "median": float(np.median(finite)),
        "std": float(finite.std()), "p5": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
    }


def aggregate(
    stats: np.ndarray, sources: np.ndarray, source_names: dict[int, str]
) -> dict[str, Any]:
    """Pooled and per-source summaries, plus the macro-over-sources Dice used for selection."""
    result: dict[str, Any] = {"overall": {}, "by_source": {}}
    scores = scores_from_stats(stats)
    for key, values in scores.items():
        result["overall"][key] = summarise(values)
    per_source_dice = []
    for source_id, name in source_names.items():
        mask = sources == source_id
        if not mask.any():
            continue
        result["by_source"][name] = {key: summarise(values[mask]) for key, values in scores.items()}
        per_source_dice.append(float(scores["dice"][mask].mean()))
    result["macro_dice"] = float(np.mean(per_source_dice)) if per_source_dice else float("nan")
    result["pooled_dice"] = float(scores["dice"].mean()) if len(stats) else float("nan")
    return result


def by_size_bin(stats: np.ndarray, gt_fraction: np.ndarray) -> dict[str, Any]:
    scores = scores_from_stats(stats)
    out: dict[str, Any] = {}
    for label, low, high in SIZE_BINS:
        mask = (gt_fraction >= low) & (gt_fraction < high)
        out[label] = {
            "n": int(mask.sum()),
            "dice": summarise(scores["dice"][mask]),
            "recall": summarise(scores["recall"][mask]),
            "precision": summarise(scores["precision"][mask]),
        }
    return out


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile symmetric surface distance, in pixels."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float("nan")

    def surface(mask: np.ndarray) -> np.ndarray:
        return mask & ~ndimage.binary_erosion(mask, border_value=0)

    pred_surface, gt_surface = surface(pred), surface(gt)
    to_gt = ndimage.distance_transform_edt(~gt_surface)
    to_pred = ndimage.distance_transform_edt(~pred_surface)
    distances = np.concatenate([to_gt[pred_surface], to_pred[gt_surface]])
    return float(np.percentile(distances, 95))


def expected_calibration_error(
    probs: np.ndarray, truth: np.ndarray, bins: int = 15
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Pixel-wise ECE. Returns ``(ece, bin_confidence, bin_accuracy, bin_weight)``."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)
    confidence = np.zeros(bins)
    accuracy = np.zeros(bins)
    weight = np.zeros(bins)
    total = max(1, probs.size)
    for b in range(bins):
        sel = index == b
        count = int(sel.sum())
        if count:
            confidence[b] = float(probs[sel].mean())
            accuracy[b] = float(truth[sel].mean())
            weight[b] = count / total
    ece = float((weight * np.abs(confidence - accuracy)).sum())
    return ece, confidence, accuracy, weight
