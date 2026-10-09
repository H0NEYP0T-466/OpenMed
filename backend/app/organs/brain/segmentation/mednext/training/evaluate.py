"""Post-training stages: post-processing search (validation) and the final test.

Validation selects the weights *and* the post-processing (threshold, minimum
component area, flip-TTA). The test split is evaluated exactly once, on the
chosen configuration, through the **serving** pipeline: raw files are decoded,
resized, segmented, resampled back to their original resolution and compared
with the original-resolution masks. The number reported is therefore the number
the application would produce, not a proxy computed on cached canvases.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import cv2
import numpy as np
import torch

from ..pipeline import MedNeXtSegmenter, postprocess_mask
from ..preprocessor import decode_gray, to_canvas
from .audit import CheckLog
from .config import CHECKPOINT_NAME, TrainConfig
from .data import SOURCE_IDS, ArrayCache
from .export import build_serving_checkpoint, plain, save_serving_checkpoint
from .metrics import (
    aggregate,
    by_size_bin,
    expected_calibration_error,
    hd95,
    scores_from_stats,
    summarise,
)
from .splits import Record, SplitManifest

logger = logging.getLogger("evaluate")

SOURCE_NAMES = {v: k for k, v in SOURCE_IDS.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Validation: post-processing search
# ─────────────────────────────────────────────────────────────────────────────

def _canvas_probabilities(
    segmenter: MedNeXtSegmenter, cache: ArrayCache, rows: np.ndarray, tta: bool, batch: int = 32
) -> np.ndarray:
    """Canvas-space probabilities for cached rows, float16 ``(N, S, S)``."""
    out = np.zeros((len(rows), cache.size, cache.size), dtype=np.float16)
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        canvases = [np.array(cache.images[r]) for r in chunk]
        probs = segmenter._forward(canvases)
        if tta:
            flipped = segmenter._forward([c[:, ::-1].copy() for c in canvases])
            probs = 0.5 * (probs + flipped[:, :, ::-1])
        out[start:start + len(chunk)] = probs.astype(np.float16)
    return out


def _stats_for(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int((pred & gt).sum())
    return np.array([tp, int((pred & ~gt).sum()), int((~pred & gt).sum()),
                     int(pred.sum()), int(gt.sum())], dtype=np.float64)


def tune_postprocess(
    segmenter: MedNeXtSegmenter,
    cache: ArrayCache,
    val_records: list[Record],
    cfg: TrainConfig,
) -> dict[str, Any]:
    """Grid-search threshold x min-area x TTA on the validation split."""
    rows = np.array([cache.row(r) for r in val_records])
    sources = np.array([SOURCE_IDS.get(r.source, 0) for r in val_records])
    truth = np.stack([np.array(cache.masks[r]) for r in rows])

    def macro(stats: np.ndarray) -> float:
        dice = scores_from_stats(stats)["dice"]
        return float(np.mean([dice[sources == s].mean() for s in np.unique(sources)]))

    results: dict[str, Any] = {"grid": {}, "tta": {}}
    best: Optional[dict[str, Any]] = None
    baseline_score: Optional[float] = None

    for tta in ((False, True) if cfg.tta_hflip else (False,)):
        probs = _canvas_probabilities(segmenter, cache, rows, tta)
        grid: dict[str, float] = {}
        for threshold in cfg.pp_thresholds:
            for frac in cfg.pp_min_area_fracs:
                stats = np.stack([
                    _stats_for(postprocess_mask(probs[i].astype(np.float32), threshold, frac), truth[i])
                    for i in range(len(rows))
                ])
                score = macro(stats)
                grid[f"{threshold:.2f}|{frac:.4f}"] = score
                if tta is False and threshold == 0.5 and frac == 0.0:
                    baseline_score = score
                candidate = {"threshold": threshold, "min_area_frac": frac,
                             "tta_hflip": tta, "score": score}
                if best is None or score > best["score"] + (0.0005 if tta else 0.0):
                    best = candidate
        results["grid"]["tta" if tta else "plain"] = grid
        results["tta"]["tta" if tta else "plain"] = max(grid.values())
        logger.info("post-processing search (tta=%s): best %.4f over %d settings", tta, max(grid.values()), len(grid))

    assert best is not None
    chosen = {
        "threshold": float(best["threshold"]), "min_area_frac": float(best["min_area_frac"]),
        "tta_hflip": bool(best["tta_hflip"]), "tuned_on_validation": True,
        "val_macro_dice_default": baseline_score, "val_macro_dice_tuned": best["score"],
        "thresholds": list(cfg.pp_thresholds), "min_area_fracs": list(cfg.pp_min_area_fracs),
    }
    results["chosen"] = chosen
    logger.info(
        "chosen post-processing: threshold %.2f, min_area %.4f, tta %s | val macro Dice %.4f -> %.4f",
        chosen["threshold"], chosen["min_area_frac"], chosen["tta_hflip"],
        baseline_score if baseline_score is not None else float("nan"), best["score"],
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Test: through the serving pipeline, at original resolution
# ─────────────────────────────────────────────────────────────────────────────

def parity_check(
    segmenter: MedNeXtSegmenter, cache: ArrayCache, records: list[Record], log: CheckLog, n: int = 50
) -> None:
    """H17: bytes -> serving decode/canvas must equal the cached training canvas."""
    mismatched = 0
    checked = records[:n]
    for rec in checked:
        with open(rec.image_path, "rb") as handle:
            gray = decode_gray(handle.read())
        canvas, _ = to_canvas(gray, cache.size)
        if not np.array_equal(canvas, np.array(cache.images[cache.row(rec)])):
            mismatched += 1
    log.add("H17", "train/serve parity", "PASS" if not mismatched else "FAIL",
            f"{len(checked)} test files decoded from raw bytes through the serving path; "
            f"{mismatched} differ from the canvas training saw")
    if mismatched:
        raise RuntimeError("Serving preprocessing does not reproduce the training canvases.")


def evaluate_test(
    segmenter: MedNeXtSegmenter,
    cache: ArrayCache,
    records: list[Record],
    out_dir: str,
    *,
    tta: Optional[bool] = None,
    label: str = "test",
    calibration_samples: int = 4096,
    seed: int = 42,
) -> dict[str, Any]:
    """Segment every held-out record through the serving path and score it."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    stats_all, sources_all, gt_frac = [], [], []
    cal_p: list[np.ndarray] = []
    cal_y: list[np.ndarray] = []
    canvas_dice: list[float] = []

    for i, rec in enumerate(records):
        with open(rec.image_path, "rb") as handle:
            gray = decode_gray(handle.read())
        truth_raw = cv2.imread(rec.mask_path, cv2.IMREAD_UNCHANGED)
        truth = ((truth_raw[:, :, 0] if truth_raw.ndim == 3 else truth_raw) >= 128).astype(np.uint8)

        result = segmenter.predict_gray(gray, tta=tta)
        stats = _stats_for(result.mask, truth)
        score = scores_from_stats(stats[None])
        edt = hd95(result.mask, truth)

        canvas_pred = postprocess_mask(
            cv2.resize(result.probability, (cache.size, cache.size), interpolation=cv2.INTER_LINEAR)
            if result.probability.shape != (cache.size, cache.size) else result.probability,
            result.threshold, result.min_area_frac,
        )
        cached_truth = np.array(cache.masks[cache.row(rec)])
        canvas_dice.append(float(scores_from_stats(_stats_for(canvas_pred, cached_truth)[None])["dice"][0]))

        pick = rng.integers(0, truth.size, calibration_samples)
        cal_p.append(result.probability.ravel()[pick])
        cal_y.append(truth.ravel()[pick])

        rows.append({
            "source": rec.source, "stem": rec.stem, "group": rec.group,
            "gt_fraction": float(truth.mean()), "pred_fraction": float(result.mask.mean()),
            "dice": float(score["dice"][0]), "iou": float(score["iou"][0]),
            "precision": float(score["precision"][0]), "recall": float(score["recall"][0]),
            "hd95_px": edt, "n_components": result.n_components,
            "mean_probability": result.mean_probability, "peak_probability": result.peak_probability,
            "inference_ms": result.inference_ms,
        })
        stats_all.append(stats)
        sources_all.append(SOURCE_IDS.get(rec.source, 0))
        gt_frac.append(float(truth.mean()))
        if (i + 1) % 200 == 0:
            logger.info("%s: %d / %d scored", label, i + 1, len(records))

    stats_arr = np.stack(stats_all)
    sources_arr = np.array(sources_all)
    summary = aggregate(stats_arr, sources_arr, SOURCE_NAMES)
    hd = np.array([r["hd95_px"] for r in rows])
    summary["overall"]["hd95_px"] = summarise(hd)
    summary["hd95_undefined"] = int(np.isnan(hd).sum())
    for name, sid in SOURCE_IDS.items():
        if name in summary["by_source"]:
            summary["by_source"][name]["hd95_px"] = summarise(hd[sources_arr == sid])
    summary["by_size"] = by_size_bin(stats_arr, np.array(gt_frac))
    summary["missed_lesions"] = int(sum(1 for r in rows if r["pred_fraction"] == 0.0))
    summary["inference_ms_median"] = float(np.median([r["inference_ms"] for r in rows]))

    probs_flat, truth_flat = np.concatenate(cal_p), np.concatenate(cal_y)
    ece, conf, acc, weight = expected_calibration_error(probs_flat, truth_flat)
    summary["calibration"] = {"ece": ece, "bin_confidence": conf.tolist(),
                              "bin_accuracy": acc.tolist(), "bin_weight": weight.tolist()}
    summary["serve_vs_canvas"] = {
        "mean_abs_dice_difference": float(np.mean(np.abs(
            np.array([r["dice"] for r in rows]) - np.array(canvas_dice)))),
        "note": "original-resolution (serving) Dice vs the same prediction scored at 256px",
    }
    summary["worst"] = [
        {"stem": r["stem"], "source": r["source"], "dice": r["dice"]}
        for r in sorted(rows, key=lambda r: r["dice"])[:15]
    ]
    summary["config"] = {"threshold": result.threshold, "min_area_frac": result.min_area_frac,
                         "tta": result.tta}

    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    with open(os.path.join(out_dir, "metrics", f"{label}_per_image.json"), "w") as handle:
        json.dump(plain(rows), handle)
    with open(os.path.join(out_dir, "metrics", f"{label}_calibration_pixels.npz"), "wb") as handle:
        np.savez_compressed(handle, probs=probs_flat.astype(np.float16), truth=truth_flat)
    logger.info(
        "%s: Dice mean %.4f median %.4f | IoU %.4f | HD95 median %.1f px | missed %d / %d | ECE %.4f",
        label, summary["overall"]["dice"]["mean"], summary["overall"]["dice"]["median"],
        summary["overall"]["iou"]["mean"], summary["overall"]["hd95_px"]["median"] or float("nan"),
        summary["missed_lesions"], len(records), ece,
    )
    return {"summary": summary, "rows": rows}


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def finalise(
    cfg: TrainConfig, manifest: SplitManifest, cache: ArrayCache, log: CheckLog,
) -> dict[str, Any]:
    """Tune on validation, evaluate on test once, write the final serving checkpoint."""
    best_path = os.path.join(cfg.out_dir, "best", CHECKPOINT_NAME)
    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"No best checkpoint at {best_path}; training produced nothing to evaluate.")

    device = "cuda" if torch.cuda.is_available() and cfg.gpus > 0 else "cpu"
    segmenter = MedNeXtSegmenter(best_path, device=device)
    parity_check(segmenter, cache, manifest.test or manifest.val, log)

    tuning = tune_postprocess(segmenter, cache, manifest.val, cfg)
    chosen = tuning["chosen"]
    segmenter.postprocess.update(chosen)

    test_primary = evaluate_test(
        segmenter, cache, manifest.test, cfg.out_dir, label="test", seed=cfg.seed
    ) if manifest.test else None
    test_plain = evaluate_test(
        segmenter, cache, manifest.test, cfg.out_dir, tta=False, label="test_no_tta", seed=cfg.seed
    ) if manifest.test and chosen["tta_hflip"] else None
    val_final = evaluate_test(
        segmenter, cache, manifest.val, cfg.out_dir, label="val", seed=cfg.seed
    )

    payload = torch.load(best_path, map_location="cpu", weights_only=True)
    spec_state = payload["state_dict"]
    from ..model import MedNeXtSpec

    metrics = {
        "val": val_final["summary"],
        "test": test_primary["summary"] if test_primary else None,
        "test_without_tta": test_plain["summary"] if test_plain else None,
        "metric_note": ("Dice/IoU/HD95 at original resolution through the serving pipeline; "
                        "HD95 in pixels (no validated spacing)."),
    }
    training = dict(payload["training"])
    training["stage"] = "final"
    final = build_serving_checkpoint(
        spec_state, MedNeXtSpec.from_dict(payload["spec"]), cfg.image_size,
        postprocess=chosen, metrics=metrics, training=training,
    )
    final_path = os.path.join(cfg.out_dir, "best", CHECKPOINT_NAME)
    save_serving_checkpoint(final, final_path)
    logger.info("final serving checkpoint written: %s (%.1f MB)", final_path,
                os.path.getsize(final_path) / 2**20)

    reloaded = MedNeXtSegmenter(final_path, device="cpu")
    sample = manifest.test[0] if manifest.test else manifest.val[0]
    with open(sample.image_path, "rb") as handle:
        check = reloaded.predict_bytes(handle.read())[0]
    log.add("H18", "export round-trip", "PASS",
            f"final checkpoint reloads strictly on CPU and segments {sample.stem} "
            f"({check.foreground_px} foreground px)")

    summary = {
        "tuning": {k: v for k, v in tuning.items() if k != "grid"},
        "test": test_primary["summary"] if test_primary else None,
        "test_no_tta": test_plain["summary"] if test_plain else None,
        "val": val_final["summary"],
    }
    with open(os.path.join(cfg.out_dir, "metrics", "summary.json"), "w") as handle:
        json.dump(plain(summary), handle, indent=1)
    with open(os.path.join(cfg.out_dir, "metrics", "postprocess_grid.json"), "w") as handle:
        json.dump(plain(tuning["grid"]), handle)
    return {"summary": summary, "tuning": tuning, "test_rows": test_primary["rows"] if test_primary else [],
            "val_rows": val_final["rows"]}
