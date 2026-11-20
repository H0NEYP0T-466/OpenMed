"""Segmentation metrics.

Two tiers: cheap per-image TP/FP/FN counts on the accelerator for every validation
epoch, and Hausdorff-95 on the CPU for the final test only.

HD95 is in pixels (no validated pixel spacing exists) and is ``nan`` when exactly one
of prediction and truth is empty. Substituting a large finite number would look like
a measurement.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy import ndimage

# Opacity lesion size, as a fraction of the (ground-truth) lung field.
SIZE_BINS = (
    ("small  (<5% of lung)", 0.0, 0.05),
    ("medium (5-15%)", 0.05, 0.15),
    ("large  (15-30%)", 0.15, 0.30),
    ("extensive (>=30%)", 0.30, 10.0),
)

# A Normal image counts as a false positive when its predicted opacity covers at least
# this fraction of the crop, during training-time validation only. The final test uses
# "any pixel after post-processing" instead, because post-processing is tuned by then.
VAL_FP_MIN_FRACTION = 0.001


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


def stats_from_masks(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred, gt = pred.astype(bool), gt.astype(bool)
    return np.array([(pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum(),
                     pred.sum(), gt.sum()], dtype=np.float64)


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
    values = np.asarray(values, dtype=np.float64)
    empty = {"n": int(values.size), "n_finite": 0, "mean": None, "median": None, "std": None,
             "p5": None, "p95": None}
    if values.size == 0:
        return empty
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return empty
    return {
        "n": int(values.size), "n_finite": int(finite.size),
        "mean": float(finite.mean()), "median": float(np.median(finite)),
        "std": float(finite.std()), "p5": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
    }


def by_size_bin(dice: np.ndarray, gt_fraction_of_lung: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, low, high in SIZE_BINS:
        mask = (gt_fraction_of_lung >= low) & (gt_fraction_of_lung < high)
        out[label] = {"n": int(mask.sum()), "dice": summarise(dice[mask])}
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
    confidence, accuracy, weight = np.zeros(bins), np.zeros(bins), np.zeros(bins)
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
