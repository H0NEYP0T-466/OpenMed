"""Post-training stages: post-processing search (validation) and the final test.

Validation selects the weights **and** the post-processing. The test split is
evaluated exactly once, on the chosen configuration, through the **serving**
pipeline: the raw PNG is decoded with the training-time decoder, segmented by
:class:`~..pipeline.LungSegmenter`, and compared against the original-resolution
masks. The number reported is therefore the number the application produces.

Two searches, run in order, because head B's crop is cut from head A's *predicted*
lung field:

    lung      threshold x minimum component area, keep-largest-k, hole filling,
              flip-TTA on/off -- scored by mean lung Dice on validation
    opacity   threshold x minimum component area, flip-TTA on/off -- scored as
              positive Dice - fpr_weight * Normal false-positive rate

The score deliberately subtracts Normal false positives: a head that paints opacity
everywhere would otherwise win on Dice alone. Non-COVID is never scored (its mask is
an annotation negative); its behaviour is reported as a line, not a number.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any

import numpy as np
from scipy import ndimage

from ..checkpoint import (
    build_serving_checkpoint,
    load_serving_checkpoint,
    plain,
    save_serving_checkpoint,
)
from ..labels import CLASS_COVID, CLASS_NON_COVID, CLASS_NORMAL, NON_COVID_NOTE
from ..model import SegSpec
from ..pipeline import LungSegmenter
from ..postprocess import (
    LungPost,
    OpacityPost,
    component_areas,
    postprocess_lung,
)
from ..preprocessor import (
    crop_resize_image,
    decode_gray,
    full_box,
    lung_crop_box,
    read_mask,
    uncrop_plane,
)
from .config import TrainConfig
from .data import ArrayCache
from .hygiene import CheckLog
from .metrics import (
    by_size_bin,
    expected_calibration_error,
    hd95,
    scores_from_stats,
    stats_from_masks,
    summarise,
)
from .splits import Record, SplitResult

logger = logging.getLogger("evaluate")

SPLIT_NOTE = (
    "Dice, IoU and HD95 at original resolution through the serving pipeline. HD95 is in "
    "pixels: no validated pixel spacing exists for this dataset. Opacity Dice is computed "
    "on COVID-19 positives only."
)

CALIBRATION_PIXELS = 4096


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _ttas(cfg: TrainConfig) -> tuple[bool, ...]:
    return (False, True) if cfg.tta_hflip else (False,)


def _key(tta: bool) -> str:
    return "tta" if tta else "plain"


def _tune_sample(records: list[Record], limit: int) -> list[Record]:
    """A deterministic, class-proportional stride through the manifest order.

    ``split_manifest.csv`` lists each class contiguously, so a stride samples every
    class in proportion instead of favouring whichever class happens to be first.
    """
    if limit <= 0 or len(records) <= limit:
        return list(records)
    index = np.linspace(0, len(records) - 1, limit).round().astype(int)
    return [records[i] for i in dict.fromkeys(index.tolist())]


def _probabilities(
    segmenter: LungSegmenter, canvases: list[np.ndarray], head: str, tta: bool
) -> np.ndarray:
    """Batch probabilities on 256 px canvases, flip-TTA merged.

    Flip-TTA is legal here: both heads are binary and bilateral, so a horizontal
    flip is an involution and needs no label swap.
    """
    probs = segmenter._probabilities(canvases, head)
    if tta:
        flipped = segmenter._probabilities(
            [np.ascontiguousarray(canvas[:, ::-1]) for canvas in canvases], head
        )
        probs = 0.5 * (probs + flipped[:, :, ::-1])
    return np.ascontiguousarray(probs, dtype=np.float32)


def mask_from_components(
    labels: np.ndarray, areas: np.ndarray, min_area_frac: float, keep_largest_k: int,
    fill_holes: bool,
) -> np.ndarray:
    """``postprocess_lung`` computed from pre-labelled components.

    The grid search labels each threshold once and then evaluates every minimum-area
    setting from those labels, which is what makes a 45-cell search affordable. This
    function must stay byte-identical to :func:`postprocess_lung`; a unit test holds
    the two together.
    """
    if areas.size == 0:
        return np.zeros(labels.shape, dtype=np.uint8)
    minimum = max(1, round(min_area_frac * labels.size)) if min_area_frac > 0 else 1
    keep = [i + 1 for i, area in enumerate(areas) if area >= minimum]
    if not keep:
        keep = [1 + int(np.argmax(areas))]
    if keep_largest_k > 0 and len(keep) > keep_largest_k:
        keep = sorted(keep, key=lambda index: areas[index - 1], reverse=True)[:keep_largest_k]
    mask = np.isin(labels, keep)
    if fill_holes and mask.any():
        mask = ndimage.binary_fill_holes(mask)
    return mask.astype(np.uint8)


def _mean_dice(stats: np.ndarray) -> float:
    if stats.shape[0] == 0:
        return float("nan")
    return float(np.mean(scores_from_stats(stats)["dice"]))


def _grid_dict(scores: dict[tuple[float, float], float]) -> dict[str, float]:
    return {f"{threshold:.2f}|{frac:.4f}": score for (threshold, frac), score in scores.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Validation: lung post-processing
# ─────────────────────────────────────────────────────────────────────────────


def tune_lung(
    segmenter: LungSegmenter, cache: ArrayCache, records: list[Record], cfg: TrainConfig
) -> dict[str, Any]:
    """Grid-search the lung head on validation, then refine keep-k and hole filling."""
    sample = _tune_sample(records, cfg.pp_tune_max_images)
    if not sample:
        raise RuntimeError("No validation records to tune the lung post-processing on.")
    thresholds = list(cfg.pp_thresholds)
    fracs = list(cfg.pp_lung_min_area_fracs)
    stats: dict[tuple[str, float, float], np.ndarray] = {
        (_key(tta), threshold, frac): np.zeros((len(sample), 5), dtype=np.float64)
        for tta in _ttas(cfg)
        for threshold in thresholds
        for frac in fracs
    }
    logger.info(
        "lung grid: %d settings x %d images x %d TTA variants", len(thresholds) * len(fracs),
        len(sample), len(_ttas(cfg)),
    )

    # One forward pass per TTA variant; every cell of the grid reads the same probs.
    for tta in _ttas(cfg):
        for start in range(0, len(sample), 16):
            chunk = sample[start:start + 16]
            canvases = [cache.image(record.uid) for record in chunk]
            probs = _probabilities(segmenter, canvases, "lung", tta)
            truths = [cache.lung(record.uid) for record in chunk]
            for threshold in thresholds:
                components = [component_areas(prob >= threshold) for prob in probs]
                for frac in fracs:
                    target = stats[(_key(tta), threshold, frac)]
                    for offset, (_, labels, areas) in enumerate(components):
                        mask = mask_from_components(labels, areas, frac, 2, True)
                        target[start + offset] = stats_from_masks(mask, truths[offset])

    scores = {key: _mean_dice(value) for key, value in stats.items()}
    best_key = max(scores, key=lambda key: scores[key])
    best_tta, best_threshold, best_frac = best_key
    best_tta_bool = best_tta == "tta"
    logger.info(
        "lung grid best: threshold %.2f, min-area %.4f, tta %s -> Dice %.4f",
        best_threshold, best_frac, best_tta_bool, scores[best_key],
    )

    # Refine the categorical knobs at the winning threshold / area: keep-largest-k and
    # hole filling. One forward pass; every variant reads the same probabilities.
    variants = [(keep_k, fill) for keep_k in cfg.pp_keep_k for fill in (True, False)]
    collected: dict[tuple[int, bool], list[np.ndarray]] = {variant: [] for variant in variants}
    for start in range(0, len(sample), 16):
        chunk = sample[start:start + 16]
        canvases = [cache.image(record.uid) for record in chunk]
        probs = _probabilities(segmenter, canvases, "lung", best_tta_bool)
        truths = [cache.lung(record.uid) for record in chunk]
        for prob, truth in zip(probs, truths, strict=True):
            _, labels, areas = component_areas(prob >= best_threshold)
            for keep_k, fill in variants:
                mask = mask_from_components(labels, areas, best_frac, keep_k, fill)
                collected[(keep_k, fill)].append(stats_from_masks(mask, truth))
    refine_scores = {
        (keep_k, fill): _mean_dice(np.stack(values))
        for (keep_k, fill), values in collected.items()
    }
    refine = {
        f"keep_k={keep_k}|fill_holes={fill}": score
        for (keep_k, fill), score in refine_scores.items()
    }
    chosen = {
        "threshold": float(best_threshold), "min_area_frac": float(best_frac),
        "keep_largest_k": 2, "fill_holes": True, "tta_hflip": best_tta_bool,
    }
    best_score = scores[best_key]
    for (keep_k, fill), score in refine_scores.items():
        if score > best_score + 1e-6:
            best_score = score
            chosen.update({"keep_largest_k": keep_k, "fill_holes": fill})
    logger.info(
        "lung refinement: keep-largest-k %d, fill holes %s -> Dice %.4f (grid %.4f)",
        chosen["keep_largest_k"], chosen["fill_holes"], best_score,
        scores[best_key],
    )

    baseline = scores.get(("plain", 0.5, 0.0))
    chosen.update({
        "tuned_on_validation": True, "grid_sample": len(sample),
        "val_lung_dice_default": baseline, "val_lung_dice_tuned": best_score,
    })
    grid = {
        key: {f"{threshold:.2f}|{frac:.4f}": scores[(key, threshold, frac)]
              for threshold in thresholds for frac in fracs}
        for key in {_key(tta) for tta in _ttas(cfg)}
    }
    return {"chosen": chosen, "grid": grid, "thresholds": thresholds,
            "min_area_fracs": fracs, "refine": refine}


# ─────────────────────────────────────────────────────────────────────────────
# Validation: opacity post-processing
# ─────────────────────────────────────────────────────────────────────────────


def _supervised(records: list[Record]) -> list[Record]:
    return [record for record in records if record.opacity_supervised]


def tune_opacity(
    segmenter: LungSegmenter, cache: ArrayCache, records: list[Record], cfg: TrainConfig,
    lung_post: LungPost,
) -> dict[str, Any]:
    """Grid-search head B on the supervised validation subset.

    The crop is cut from the *predicted* lung field, exactly as serving does, so this
    runs after the lung post-processing has been chosen.
    """
    sample = _tune_sample(_supervised(records), cfg.pp_tune_max_images)
    if not sample:
        raise RuntimeError("No opacity-supervised validation records; head B cannot be tuned.")
    thresholds = list(cfg.pp_thresholds)
    fracs = list(cfg.pp_opacity_min_area_fracs)
    positives = np.array([record.cls == CLASS_COVID for record in sample])

    # Per-(tta, threshold, frac) counts: positive images get 5 stats, Normal images
    # contribute a single false-positive flag and a predicted-area fraction.
    dice_stats: dict[tuple[str, float, float], np.ndarray] = {
        (_key(tta), threshold, frac): np.zeros((len(sample), 5), dtype=np.float64)
        for tta in _ttas(cfg) for threshold in thresholds for frac in fracs
    }
    normal_fp: dict[tuple[str, float, float], np.ndarray] = {
        key: np.zeros(len(sample), dtype=bool) for key in dice_stats
    }
    normal_frac: dict[tuple[str, float, float], np.ndarray] = {
        key: np.zeros(len(sample), dtype=np.float64) for key in dice_stats
    }
    logger.info(
        "opacity grid: %d settings x %d supervised images x %d TTA variants",
        len(thresholds) * len(fracs), len(sample), len(_ttas(cfg)),
    )

    for tta in _ttas(cfg):
        for start in range(0, len(sample), 16):
            chunk = sample[start:start + 16]
            canvases = [cache.image(record.uid) for record in chunk]
            lung_probs = _probabilities(segmenter, canvases, "lung", lung_post.tta_hflip)
            lung_masks = [postprocess_lung(prob, lung_post) for prob in lung_probs]
            boxes = [lung_crop_box(mask) or full_box(mask.shape) for mask in lung_masks]
            crops = [crop_resize_image(canvas, box)
                     for canvas, box in zip(canvases, boxes, strict=True)]
            crop_probs = _probabilities(segmenter, crops, "opacity", tta)
            probs = [uncrop_plane(prob, box, canvas.shape)
                     for prob, box, canvas in zip(crop_probs, boxes, canvases, strict=True)]

            for threshold in thresholds:
                components = [component_areas(prob >= threshold) for prob in probs]
                for frac in fracs:
                    key = (_key(tta), threshold, frac)
                    for offset, (lung_mask, record) in enumerate(
                        zip(lung_masks, chunk, strict=True)
                    ):
                        _, labels, areas = components[offset]
                        mask = _opacity_from_components(labels, areas, frac) & lung_mask
                        area = int(mask.sum())
                        lung_area = max(1, int(lung_mask.sum()))
                        normal_frac[key][start + offset] = area / lung_area
                        if record.cls == CLASS_COVID:
                            truth = cache.infection(record.uid)
                            assert truth is not None
                            dice_stats[key][start + offset] = stats_from_masks(mask, truth)
                        elif record.cls == CLASS_NORMAL:
                            normal_fp[key][start + offset] = area > 0
                        # Non-COVID never enters the search: its mask is not a target.

    def score(key: tuple[str, float, float]) -> tuple[float, float, float]:
        positive_dice = _mean_dice(dice_stats[key][positives])
        negatives = ~positives
        fpr = float(normal_fp[key][negatives].mean()) if negatives.any() else 0.0
        return positive_dice - cfg.fpr_weight * fpr, positive_dice, fpr

    scored = {key: score(key) for key in dice_stats}
    best_key = max(scored, key=lambda key: scored[key][0])
    best_tta, best_threshold, best_frac = best_key
    best_tta_bool = best_tta == "tta"
    logger.info(
        "opacity grid best: threshold %.2f, min-area %.4f, tta %s -> positive Dice %.4f, "
        "Normal FPR %.4f, score %.4f", best_threshold, best_frac, best_tta_bool,
        scored[best_key][1], scored[best_key][2], scored[best_key][0],
    )
    baseline = scored.get(("plain", 0.5, 0.0))
    negatives = ~positives
    chosen = {
        "threshold": float(best_threshold), "min_area_frac": float(best_frac),
        "tta_hflip": best_tta_bool, "tuned_on_validation": True,
        "grid_sample": len(sample),
        "val_normal_mean_predicted_fraction": (
            float(normal_frac[best_key][negatives].mean()) if negatives.any() else None
        ),
        "val_positive_dice_default": baseline[1] if baseline else None,
        "val_normal_fpr_default": baseline[2] if baseline else None,
        "val_positive_dice_tuned": scored[best_key][1],
        "val_normal_fpr_tuned": scored[best_key][2],
    }
    grid = {
        key: {f"{threshold:.2f}|{frac:.4f}": scored[(key, threshold, frac)][0]
              for threshold in thresholds for frac in fracs}
        for key in {_key(tta) for tta in _ttas(cfg)}
    }
    return {"chosen": chosen, "grid": grid, "thresholds": thresholds,
            "min_area_fracs": fracs}


def _opacity_from_components(
    labels: np.ndarray, areas: np.ndarray, min_area_frac: float
) -> np.ndarray:
    """``postprocess_opacity`` without the lung clip, from pre-labelled components.

    ``postprocess_opacity`` drops a component only when it is below the minimum area;
    unlike the lung mask it does not keep a survivor when everything would vanish.
    """
    if areas.size == 0:
        return np.zeros(labels.shape, dtype=np.uint8)
    minimum = max(1, round(min_area_frac * labels.size)) if min_area_frac > 0 else 1
    keep = [i + 1 for i, area in enumerate(areas) if area >= minimum]
    if not keep:
        return np.zeros(labels.shape, dtype=np.uint8)
    return np.isin(labels, keep).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# The held-out pass, through the serving pipeline
# ─────────────────────────────────────────────────────────────────────────────


def _summaries(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = np.array([row[key] for row in rows if row[key] is not None], dtype=np.float64)
    return summarise(values)


def _lung_field(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, Any] = {}
    for name in (CLASS_COVID, CLASS_NON_COVID, CLASS_NORMAL):
        subset = [row for row in rows if row["class"] == name]
        if subset:
            by_class[name] = {
                "n": len(subset),
                "dice": summarise(np.array([row["lung_dice"] for row in subset])),
            }
    return {
        "dice": _summaries(rows, "lung_dice"),
        "iou": _summaries(rows, "lung_iou"),
        "hd95_px": _summaries(rows, "lung_hd95_px"),
        "by_class": by_class,
    }


def _opacity(rows: list[dict[str, Any]], fpr_weight: float) -> dict[str, Any] | None:
    positive = [row for row in rows if row["opacity_dice"] is not None]
    normal = [row for row in rows if row["class"] == CLASS_NORMAL and row["opacity_evaluated"]]
    non_covid = [row for row in rows if row["class"] == CLASS_NON_COVID and row["opacity_evaluated"]]
    if not positive:
        return None
    dice = np.array([row["opacity_dice"] for row in positive], dtype=np.float64)
    iou = np.array([row["opacity_iou"] for row in positive], dtype=np.float64)
    hd = np.array([row["opacity_hd95_px"] for row in positive], dtype=np.float64)
    fractions = np.array([row["opacity_gt_fraction_of_lung"] for row in positive], dtype=np.float64)
    if normal:
        with_prediction = sum(1 for row in normal if row["opacity_pred_px"])
        false_positive_rate = with_prediction / len(normal)
        mean_fraction = float(np.mean([row["opacity_pred_fraction_of_lung"] for row in normal]))
    else:
        with_prediction, false_positive_rate, mean_fraction = 0, None, None
    if non_covid:
        any_rate = sum(1 for row in non_covid if row["opacity_pred_px"]) / len(non_covid)
        non_covid_fraction = float(
            np.mean([row["opacity_pred_fraction_of_lung"] for row in non_covid])
        )
    else:
        any_rate, non_covid_fraction = None, None
    return {
        "positives": {
            "n": len(positive), "dice": summarise(dice), "iou": summarise(iou),
            "hd95_px": summarise(hd), "by_size": by_size_bin(dice, fractions),
        },
        "normal": {
            "n": len(normal), "n_with_prediction": with_prediction,
            "false_positive_rate": false_positive_rate,
            "mean_predicted_fraction_of_lung": mean_fraction,
        },
        "non_covid": {
            "n": len(non_covid), "any_opacity_rate": any_rate,
            "mean_predicted_fraction_of_lung": non_covid_fraction, "note": NON_COVID_NOTE,
        },
        "selection_score": {
            "formula": "positive Dice - fpr_weight * Normal false-positive rate",
            "fpr_weight": fpr_weight,
            "value": (float(np.mean(dice)) - fpr_weight * false_positive_rate
                      if false_positive_rate is not None else None),
        },
    }


def evaluate_test(
    segmenter: LungSegmenter,
    cache: ArrayCache,
    records: list[Record],
    out_dir: str,
    *,
    label: str,
    lung_post: LungPost,
    opacity_post: OpacityPost | None,
    cfg: TrainConfig,
) -> dict[str, Any]:
    """Segment every record through the serving path and score it against the files."""
    rows: list[dict[str, Any]] = []
    calibration: list[tuple[np.ndarray, np.ndarray]] = []
    rng = np.random.default_rng(cfg.seed)

    for index, record in enumerate(records):
        with open(record.image_path, "rb") as handle:
            gray = decode_gray(handle.read())
        truth_lung = read_mask(record.lung_mask_path)
        result = segmenter.predict_gray(
            gray, lung_post=lung_post, opacity_post=opacity_post, run_opacity=True
        )
        lung_scores = scores_from_stats(stats_from_masks(result.lung_mask, truth_lung)[None])

        row: dict[str, Any] = {
            "filename": record.filename, "class": record.cls, "split": label,
            "supervised": bool(record.opacity_supervised),
            "lung_dice": float(lung_scores["dice"][0]),
            "lung_iou": float(lung_scores["iou"][0]),
            "lung_precision": float(lung_scores["precision"][0]),
            "lung_recall": float(lung_scores["recall"][0]),
            "lung_hd95_px": hd95(result.lung_mask, truth_lung),
            "lung_pred_px": int(result.lung_mask.sum()),
            "lung_gt_px": int(truth_lung.sum()),
            "lung_components": int(result.stats["lung_components"]),
            "crop_fallback": bool(result.crop_fallback),
            "inference_ms": float(result.inference_ms),
            "opacity_evaluated": False, "opacity_dice": None, "opacity_iou": None,
            "opacity_hd95_px": None, "opacity_gt_fraction_of_lung": None,
            "opacity_pred_fraction_of_lung": None, "opacity_pred_px": None,
        }

        truth_opacity = cache.infection(record.uid)
        if result.opacity_mask is not None and truth_opacity is not None:
            lung_area = max(1, int(truth_lung.sum()))
            predicted = int(result.opacity_mask.sum())
            row.update({
                "opacity_evaluated": True,
                "opacity_pred_px": predicted,
                "opacity_pred_fraction_of_lung": predicted / lung_area,
                "opacity_gt_fraction_of_lung": float(truth_opacity.sum()) / lung_area,
            })
            if record.cls == CLASS_COVID:
                scores = scores_from_stats(
                    stats_from_masks(result.opacity_mask, truth_opacity)[None]
                )
                row.update({
                    "opacity_dice": float(scores["dice"][0]),
                    "opacity_iou": float(scores["iou"][0]),
                    "opacity_hd95_px": hd95(result.opacity_mask, truth_opacity),
                })
            if result.opacity_probability is not None:
                region = result.lung_mask > 0
                flat = result.opacity_probability[region]
                truth_flat = truth_opacity[region]
                if flat.size:
                    take = min(CALIBRATION_PIXELS, flat.size)
                    pick = rng.choice(flat.size, size=take, replace=False)
                    calibration.append((flat[pick], truth_flat[pick].astype(np.float32)))

        rows.append(row)
        if (index + 1) % 200 == 0:
            logger.info("%s: %d / %d scored", label, index + 1, len(records))

    if not rows:
        raise RuntimeError(f"No records to evaluate for the {label} split.")
    summary = {
        "split": label,
        "n_images": len(rows),
        "note": SPLIT_NOTE,
        "lung_field": _lung_field(rows),
        "opacity": _opacity(rows, cfg.fpr_weight),
    }
    hd = np.array([row["lung_hd95_px"] for row in rows], dtype=np.float64)
    summary["hd95_undefined"] = int(np.isnan(hd).sum())
    summary["inference_ms_median"] = float(np.median([row["inference_ms"] for row in rows]))
    if calibration and summary["opacity"]:
        probs = np.concatenate([pair[0] for pair in calibration])
        truth = np.concatenate([pair[1] for pair in calibration])
        ece, confidence, accuracy, weight = expected_calibration_error(probs, truth)
        summary["opacity"]["calibration"] = {
            "ece": ece, "bin_confidence": confidence.tolist(),
            "bin_accuracy": accuracy.tolist(), "bin_weight": weight.tolist(),
            "pixels": int(probs.size),
            "note": "pixel reliability inside the predicted lung field",
        }
        os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
        with open(os.path.join(out_dir, "metrics", f"{label}_calibration_pixels.npz"), "wb") as handle:
            np.savez_compressed(handle, probs=probs.astype(np.float16), truth=truth.astype(np.uint8))

    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    with open(os.path.join(out_dir, "metrics", f"{label}_per_image.json"), "w", encoding="utf-8") as handle:
        json.dump(plain(rows), handle)
    with open(os.path.join(out_dir, "metrics", f"{label}_per_image.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(
        "%s: lung Dice mean %.4f median %.4f | HD95 median %s px%s",
        label, summary["lung_field"]["dice"]["mean"], summary["lung_field"]["dice"]["median"],
        f"{summary['lung_field']['hd95_px']['median']:.1f}"
        if summary["lung_field"]["hd95_px"]["median"] is not None else "n/a",
        (f" | opacity positives {summary['opacity']['positives']['n']} Dice "
         f"{summary['opacity']['positives']['dice']['mean']:.4f} | Normal FPR "
         f"{summary['opacity']['normal']['false_positive_rate']:.4f}"
         if summary["opacity"] else " | no opacity head"),
    )
    return {"summary": summary, "rows": rows}


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────


def parity_check(
    segmenter: LungSegmenter, cache: ArrayCache, records: list[Record], log: CheckLog, n: int = 40
) -> None:
    """H17: raw bytes -> serving decode must reproduce the cached training canvas."""
    mismatched = 0
    checked = records[:n]
    for record in checked:
        with open(record.image_path, "rb") as handle:
            gray = decode_gray(handle.read())
        if not np.array_equal(gray, np.array(cache.image(record.uid))):
            mismatched += 1
    log.add(
        "H17", "train/serve parity", "PASS" if not mismatched else "FAIL",
        f"{len(checked)} held-out files decoded through the serving path; {mismatched} differ "
        "from the 256 px canvas training saw",
    )
    if mismatched:
        raise RuntimeError("Serving preprocessing does not reproduce the training canvases.")


def finalise(
    cfg: TrainConfig, split: SplitResult, cache: ArrayCache, log: CheckLog,
    best_path: str, heads_trained: tuple[str, ...],
) -> dict[str, Any]:
    """Tune on validation, evaluate test once, rewrite the best checkpoint."""
    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"No best checkpoint at {best_path}; training produced nothing.")
    segmenter = LungSegmenter(best_path)
    parity_check(segmenter, cache, split.part("val"), log)

    if len(split.part("val")) < 32:
        raise RuntimeError("Validation split is too small to tune post-processing on.")

    lung_tuning = tune_lung(segmenter, cache, split.part("val"), cfg)
    lung_post = LungPost.from_dict(lung_tuning["chosen"])

    opacity_tuning: dict[str, Any] | None = None
    opacity_post: OpacityPost | None = None
    if "opacity" in heads_trained:
        opacity_tuning = tune_opacity(segmenter, cache, split.part("val"), cfg, lung_post)
        opacity_post = OpacityPost.from_dict(opacity_tuning["chosen"])

    test = evaluate_test(
        segmenter, cache, split.part("test"), cfg.out_dir, label="test",
        lung_post=lung_post, opacity_post=opacity_post, cfg=cfg,
    )
    ablation = None
    if (lung_post.tta_hflip or (opacity_post is not None and opacity_post.tta_hflip)):
        ablation = evaluate_test(
            segmenter, cache, split.part("test"), cfg.out_dir, label="test_no_tta",
            lung_post=LungPost(**{**lung_post.to_dict(), "tta_hflip": False}),
            opacity_post=(OpacityPost(**{**opacity_post.to_dict(), "tta_hflip": False})
                          if opacity_post else None),
            cfg=cfg,
        )
    val = evaluate_test(
        segmenter, cache, split.part("val"), cfg.out_dir, label="val",
        lung_post=lung_post, opacity_post=opacity_post, cfg=cfg,
    )

    postprocess = {
        "lung": {**lung_post.to_dict(), "tuned_on_validation": True},
        "opacity": ({**opacity_post.to_dict(), "tuned_on_validation": True}
                    if opacity_post is not None else None),
    }
    payload = load_serving_checkpoint(best_path)
    training = dict(payload.get("training", {}))
    training.update({"stage": "final", "phase": "final-after-test"})
    metrics = {
        "test": test["summary"],
        "val": val["summary"],
        "test_without_tta": ablation["summary"] if ablation else None,
        "metric_note": SPLIT_NOTE,
    }
    final = build_serving_checkpoint(
        payload["state_dict"], SegSpec.from_dict(payload["spec"]),
        heads_trained=tuple(payload["labels"]["heads_trained"]),
        postprocess=postprocess, metrics=metrics, training=training,
    )
    save_serving_checkpoint(final, best_path)
    logger.info("final serving checkpoint written: %s (%.1f MB)", best_path,
                os.path.getsize(best_path) / 2**20)

    segmenter.unload()
    reloaded = LungSegmenter(best_path, device="cpu")
    sample = split.part("test")[0] if split.part("test") else split.part("val")[0]
    with open(sample.image_path, "rb") as handle:
        check = reloaded.predict_bytes(handle.read())[0]
    log.add(
        "H18", "export round-trip", "PASS",
        f"final checkpoint reloads strictly on CPU and segments {sample.filename} "
        f"({int(check.lung_mask.sum())} lung px, opacity "
        f"{'available' if check.opacity_available else 'absent'})",
    )
    reloaded.unload()

    summary = {
        "tuning": {
            "lung": {key: value for key, value in lung_tuning.items() if key != "grid"},
            "opacity": ({key: value for key, value in opacity_tuning.items() if key != "grid"}
                        if opacity_tuning else None),
        },
        "test": test["summary"],
        "test_without_tta": ablation["summary"] if ablation else None,
        "val": val["summary"],
        "postprocess": postprocess,
        "heads_trained": list(heads_trained),
    }
    os.makedirs(os.path.join(cfg.out_dir, "metrics"), exist_ok=True)
    with open(os.path.join(cfg.out_dir, "metrics", "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(plain(summary), handle, indent=1)
    grids = {"lung": lung_tuning["grid"],
             "opacity": opacity_tuning["grid"] if opacity_tuning else None}
    with open(os.path.join(cfg.out_dir, "metrics", "postprocess_grid.json"), "w",
              encoding="utf-8") as handle:
        json.dump(plain(grids), handle)
    return {
        "summary": summary,
        "lung_tuning": lung_tuning,
        "opacity_tuning": opacity_tuning,
        "test_rows": test["rows"],
        "val_rows": val["rows"],
        "ablation": ablation,
    }
