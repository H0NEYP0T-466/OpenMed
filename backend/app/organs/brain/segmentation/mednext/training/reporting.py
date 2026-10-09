"""Plots, the plain-text training report, and the Kaggle export ZIP.

Every figure is saved to ``<out>/plots``. Plotting never raises into the run: a
failed figure is logged and skipped, because losing a plot must not cost the
trained model its export.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import time
import zipfile
from typing import Any, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..pipeline import MedNeXtSegmenter  # noqa: E402
from .audit import SourceAudit  # noqa: E402
from .augment import augment  # noqa: E402
from .config import CHECKPOINT_NAME, TrainConfig  # noqa: E402
from .data import ArrayCache  # noqa: E402
from .splits import SplitManifest, _numeric_order, _standardised, adjacent_similarity  # noqa: E402

logger = logging.getLogger("reporting")

INK, ACCENT, OLIVE, MUSTARD, MUTE = "#15140f", "#ed6f5c", "#6b7a3a", "#c9a227", "#8b8676"
SOURCE_COLOURS = {"openmed": ACCENT, "btsc": OLIVE}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#f7f1de", "axes.facecolor": "#f7f1de", "savefig.facecolor": "#f7f1de",
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    })


def _save(fig: plt.Figure, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, "plots", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    logger.info("plot written: %s", name)


def _safely(fn):
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - a failed figure must not abort the export
            logger.warning("plot %s skipped: %s", fn.__name__, exc)
            plt.close("all")
    wrapper.__name__ = fn.__name__
    return wrapper


def _load_json(path: str, default: Any) -> Any:
    if not os.path.isfile(path):
        return default
    with open(path) as handle:
        return json.load(handle)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

@_safely
def plot_learning_curves(history: list[dict[str, Any]], out_dir: str) -> None:
    if not history:
        return
    _style()
    epochs = [h["epoch"] for h in history]
    best = max(history, key=lambda h: h["best_score"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))

    ax = axes[0, 0]
    ax.plot(epochs, [h["train_loss"] for h in history], color=ACCENT, label="train (deep-supervised)")
    ax.plot(epochs, [h["val_loss_raw"] for h in history], color=INK, label="val raw")
    ax.plot(epochs, [h["val_loss_ema"] for h in history], color=OLIVE, label="val EMA")
    ax.set(title="Loss", xlabel="epoch", ylabel="BCE/Dice compound")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, [h["val_macro_raw"] for h in history], color=INK, label="raw")
    ax.plot(epochs, [h["val_macro_ema"] for h in history], color=OLIVE, label="EMA")
    ax.axvline(best["epoch"] if "epoch" in best else epochs[-1], color=ACCENT, ls="--", lw=1)
    ax.set(title="Validation Dice (macro over sources)", xlabel="epoch", ylim=(0, 1))
    ax.legend()

    ax = axes[0, 2]
    for name, colour in SOURCE_COLOURS.items():
        key = f"val_dice_{name}_ema"
        if key in history[0]:
            ax.plot(epochs, [h[key] for h in history], color=colour, label=name)
    ax.set(title="Validation Dice by source (EMA)", xlabel="epoch", ylim=(0, 1))
    ax.legend()

    ax = axes[1, 0]
    ax.plot(epochs, [h["lr"] for h in history], color=MUSTARD)
    ax.set(title="Learning rate (warm-up + cosine)", xlabel="epoch", yscale="log")

    ax = axes[1, 1]
    ax.plot(epochs, [h["val_precision_ema"] for h in history], color=INK, label="precision")
    ax.plot(epochs, [h["val_recall_ema"] for h in history], color=ACCENT, label="recall")
    ax.plot(epochs, [h["val_iou_ema"] for h in history], color=OLIVE, label="IoU")
    ax.set(title="Validation precision / recall / IoU (EMA)", xlabel="epoch", ylim=(0, 1))
    ax.legend()

    ax = axes[1, 2]
    ax.plot(epochs, [h["grad_norm"] for h in history], color=INK, label="grad norm")
    ax.set(title="Gradient norm and epoch time", xlabel="epoch")
    ax2 = ax.twinx()
    ax2.plot(epochs, [h["epoch_seconds"] / 60 for h in history], color=MUTE, label="min/epoch")
    ax2.set_ylabel("minutes")
    ax2.grid(False)
    _save(fig, out_dir, "01_learning_curves.png")


@_safely
def plot_dataset_overview(
    manifest: SplitManifest, audits: dict[str, SourceAudit], out_dir: str
) -> None:
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))

    ax = axes[0, 0]
    bottoms = np.zeros(3)
    for name, colour in SOURCE_COLOURS.items():
        counts = np.array([sum(1 for r in manifest.part(s) if r.source == name)
                           for s in ("train", "val", "test")])
        ax.bar(["train", "val", "test"], counts, bottom=bottoms, color=colour, label=name)
        bottoms += counts
    ax.set(title="Slices per split", ylabel="slices")
    ax.legend()

    ax = axes[0, 1]
    for name, colour in SOURCE_COLOURS.items():
        values = [r.fg_frac * 100 for r in manifest.all_records() if r.source == name]
        ax.hist(values, bins=np.linspace(0, 12, 49), alpha=0.6, color=colour, label=name, density=True)
    ax.set(title="Lesion size (% of frame)", xlabel="% of frame", ylabel="density")
    ax.legend()

    ax = axes[0, 2]
    for split, colour in (("train", INK), ("val", ACCENT), ("test", OLIVE)):
        values = [r.fg_frac * 100 for r in manifest.part(split)]
        if values:
            ax.hist(values, bins=np.linspace(0, 12, 41), histtype="step", lw=1.8, color=colour,
                    label=split, density=True)
    ax.set(title="Lesion size by split (parity)", xlabel="% of frame")
    ax.legend()

    ax = axes[1, 0]
    for name, colour in SOURCE_COLOURS.items():
        sizes: dict[str, int] = {}
        for r in manifest.all_records():
            if r.source == name:
                sizes[r.group] = sizes.get(r.group, 0) + 1
        if sizes:
            ax.hist(list(sizes.values()), bins=30, alpha=0.6, color=colour,
                    label=f"{name} ({len(sizes)} groups)")
    ax.set(title="Slices per group", xlabel="slices", ylabel="groups", yscale="log")
    ax.legend()

    ax = axes[1, 1]
    for name, colour in SOURCE_COLOURS.items():
        values = [r.gray_mean for r in manifest.all_records() if r.source == name]
        ax.hist(values, bins=40, alpha=0.6, color=colour, label=name, density=True)
    ax.set(title="Mean image intensity (domain gap)", xlabel="mean gray level")
    ax.legend()

    ax = axes[1, 2]
    labels, heights, colours = [], [], []
    for name, audit in audits.items():
        for reason, count in audit.summary["drop_reasons"].items():
            labels.append(f"{name}: {reason[:34]}")
            heights.append(count)
            colours.append(SOURCE_COLOURS[name])
    if labels:
        ax.barh(range(len(labels)), heights, color=colours)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
    ax.set(title="Files removed by hygiene", xlabel="files")
    ax.invert_yaxis()
    _save(fig, out_dir, "02_dataset_overview.png")


@_safely
def plot_btsc_adjacency(audits: dict[str, SourceAudit], manifest: SplitManifest, out_dir: str, seed: int) -> None:
    audit = next((a for a in audits.values() if a.spec.group_of is None), None)
    if audit is None:
        return
    _style()
    kept = [r.stem for r in manifest.all_records() if r.source == audit.spec.name]
    order = _numeric_order(kept)
    adjacent = adjacent_similarity(audit.thumbs, order)
    feats = _standardised(audit.thumbs, order)
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, len(order), 5000), rng.integers(0, len(order), 5000)
    random_corr = (feats[a] * feats[b]).mean(axis=1)
    split_of = {r.stem: s for s in ("train", "val", "test") for r in manifest.part(s)
                if r.source == audit.spec.name}

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    axes[0].hist(adjacent, bins=50, alpha=0.7, color=OLIVE, density=True, label="index-adjacent slices")
    axes[0].hist(random_corr, bins=50, alpha=0.6, color=ACCENT, density=True, label="random pairs")
    axes[0].set(title="BTSC: does index adjacency track scan membership?",
                xlabel="thumbnail correlation")
    axes[0].legend()

    colour_of = {"train": INK, "val": ACCENT, "test": OLIVE}
    axes[1].scatter(range(len(order)), [0] * len(order), c=[colour_of[split_of[s]] for s in order],
                    s=6, marker="|")
    crossing = [i for i in range(len(order) - 1) if split_of[order[i]] != split_of[order[i + 1]]]
    axes[1].scatter(crossing, [0] * len(crossing), c=MUSTARD, s=60, marker="v", label="split boundary")
    axes[1].set(title="BTSC split along the index (ink train, coral val, olive test)", yticks=[],
                xlabel="slice order")
    axes[1].legend()
    _save(fig, out_dir, "03_btsc_adjacency.png")


@_safely
def plot_samples(cache: ArrayCache, manifest: SplitManifest, cfg: TrainConfig, out_dir: str) -> None:
    _style()
    rng = np.random.default_rng(cfg.seed)
    chosen = []
    for name in ("openmed", "btsc"):
        pool = [r for r in manifest.train if r.source == name]
        chosen += [pool[i] for i in rng.choice(len(pool), 4, replace=False)] if len(pool) >= 4 else pool
    fig, axes = plt.subplots(2, 8, figsize=(19, 5.2))
    for ax, rec in zip(axes.ravel(), chosen, strict=False):
        row = cache.row(rec)
        image, mask = np.array(cache.images[row]), np.array(cache.masks[row])
        ax.imshow(image, cmap="gray")
        ax.contour(mask, levels=[0.5], colors=[ACCENT], linewidths=1.2)
        ax.set_title(f"{rec.source} {rec.fg_frac:.1%}", fontsize=8)
        ax.axis("off")
    _save(fig, out_dir, "04_training_samples.png")

    rec = chosen[0]
    row = cache.row(rec)
    image, mask = np.array(cache.images[row]), np.array(cache.masks[row])
    fig, axes = plt.subplots(2, 8, figsize=(19, 5.2))
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].contour(mask, levels=[0.5], colors=[ACCENT], linewidths=1.2)
    axes[0, 0].set_title("original", fontsize=8)
    for k, ax in enumerate(axes.ravel()[1:]):
        aug_rng = np.random.default_rng([cfg.seed, 0, k])
        aug_img, aug_mask = augment(image, mask, aug_rng, cfg)
        ax.imshow(aug_img, cmap="gray")
        ax.contour(aug_mask, levels=[0.5], colors=[ACCENT], linewidths=1.2)
        ax.set_title(f"augmented {k + 1}", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    _save(fig, out_dir, "05_augmentation_preview.png")


@_safely
def plot_test_distribution(rows: list[dict[str, Any]], summary: dict[str, Any], out_dir: str) -> None:
    if not rows:
        return
    _style()
    dice = np.array([r["dice"] for r in rows])
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))

    ax = axes[0, 0]
    for name, colour in SOURCE_COLOURS.items():
        values = [r["dice"] for r in rows if r["source"] == name]
        if values:
            ax.hist(values, bins=np.linspace(0, 1, 26), alpha=0.6, color=colour,
                    label=f"{name} (mean {np.mean(values):.3f})")
    ax.set(title="Test Dice per image", xlabel="Dice")
    ax.legend()

    ax = axes[0, 1]
    data = [[r["dice"] for r in rows if r["source"] == n] for n in SOURCE_COLOURS]
    box = ax.boxplot([d for d in data if d], tick_labels=[n for n, d in zip(SOURCE_COLOURS, data, strict=True) if d],
                     patch_artist=True)
    for patch, colour in zip(box["boxes"], SOURCE_COLOURS.values(), strict=False):
        patch.set_facecolor(colour)
        patch.set_alpha(0.6)
    ax.set(title="Dice by source", ylabel="Dice")

    ax = axes[0, 2]
    for name, colour in SOURCE_COLOURS.items():
        sel = [r for r in rows if r["source"] == name]
        ax.scatter([r["gt_fraction"] * 100 for r in sel], [r["dice"] for r in sel],
                   s=9, alpha=0.5, color=colour, label=name)
    ax.set(title="Dice vs lesion size", xlabel="lesion size (% of frame, log)", ylabel="Dice", xscale="log")
    ax.legend()

    ax = axes[1, 0]
    labels = list(summary["by_size"])
    means = [summary["by_size"][k]["dice"]["mean"] or 0 for k in labels]
    counts = [summary["by_size"][k]["n"] for k in labels]
    bars = ax.bar(range(len(labels)), means, color=ACCENT)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{k}\nn={c}" for k, c in zip(labels, counts, strict=True)], fontsize=8)
    for bar, value in zip(bars, means, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center")
    ax.set(title="Dice by lesion size", ylim=(0, 1))

    ax = axes[1, 1]
    ax.scatter([r["precision"] for r in rows], [r["recall"] for r in rows], s=8, alpha=0.4, color=INK)
    ax.set(title="Precision vs recall per image", xlabel="precision", ylabel="recall",
           xlim=(0, 1.02), ylim=(0, 1.02))

    ax = axes[1, 2]
    hd = np.array([r["hd95_px"] for r in rows], dtype=float)
    hd = hd[np.isfinite(hd)]
    ax.hist(hd, bins=40, color=OLIVE)
    ax.set(title=f"HD95 in pixels (median {np.median(hd):.1f}; undefined {len(rows) - len(hd)})",
           xlabel="px")
    _save(fig, out_dir, "06_test_distributions.png")


@_safely
def plot_postprocess_grid(tuning: dict[str, Any], cfg: TrainConfig, out_dir: str) -> None:
    _style()
    keys = [k for k in ("plain", "tta") if k in tuning["grid"]]
    fig, axes = plt.subplots(1, len(keys), figsize=(7.5 * len(keys), 5))
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys, strict=True):
        grid = tuning["grid"][key]
        matrix = np.array([[grid[f"{t:.2f}|{a:.4f}"] for a in cfg.pp_min_area_fracs]
                           for t in cfg.pp_thresholds])
        image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(cfg.pp_min_area_fracs)))
        ax.set_xticklabels([f"{a:.4f}" for a in cfg.pp_min_area_fracs])
        ax.set_yticks(range(len(cfg.pp_thresholds)))
        ax.set_yticklabels([f"{t:.2f}" for t in cfg.pp_thresholds])
        ax.set(title=f"Validation macro Dice ({'with' if key == 'tta' else 'without'} flip-TTA)",
               xlabel="minimum component area (fraction of frame)", ylabel="threshold")
        ax.grid(False)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax)
    _save(fig, out_dir, "07_postprocess_search.png")


@_safely
def plot_calibration(out_dir: str, summary: dict[str, Any]) -> None:
    cal = summary["calibration"]
    pixels = np.load(os.path.join(out_dir, "metrics", "test_calibration_pixels.npz"))
    probs, truth = pixels["probs"].astype(np.float32), pixels["truth"]
    from sklearn.metrics import average_precision_score, precision_recall_curve

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    conf, acc, weight = (np.array(cal[k]) for k in ("bin_confidence", "bin_accuracy", "bin_weight"))
    used = weight > 0
    ax.plot([0, 1], [0, 1], "--", color=MUTE, label="perfect")
    ax.plot(conf[used], acc[used], "o-", color=ACCENT, label=f"model (ECE {cal['ece']:.4f})")
    ax.set(title="Pixel reliability", xlabel="predicted probability", ylabel="observed foreground rate")
    ax.legend()

    ax = axes[1]
    ax.bar(range(len(weight)), weight, color=INK)
    ax.set(title="Pixel mass per probability bin (log)", yscale="log", xlabel="bin")

    ax = axes[2]
    precision, recall, _ = precision_recall_curve(truth, probs)
    ax.plot(recall, precision, color=OLIVE,
            label=f"AP {average_precision_score(truth, probs):.3f} (random pixels)")
    ax.set(title="Pixel precision-recall", xlabel="recall", ylabel="precision", ylim=(0, 1.02))
    ax.legend()
    _save(fig, out_dir, "08_calibration_pr.png")


@_safely
def plot_qualitative(
    cfg: TrainConfig, rows: list[dict[str, Any]], manifest: SplitManifest, out_dir: str
) -> None:
    best_path = os.path.join(out_dir, "best", CHECKPOINT_NAME)
    if not rows or not os.path.isfile(best_path):
        return
    _style()
    segmenter = MedNeXtSegmenter(best_path, device="cuda" if _cuda(cfg) else "cpu")
    by_stem = {(r.source, r.stem): r for r in manifest.test}
    ranked = sorted(rows, key=lambda r: r["dice"])
    n = len(ranked)
    picks = ranked[:4] + ranked[n // 2 - 2:n // 2 + 2] + ranked[-4:]
    fig, axes = plt.subplots(3, 4, figsize=(15, 11.5))
    for ax, row in zip(axes.ravel(), picks, strict=False):
        rec = by_stem[(row["source"], row["stem"])]
        gray = cv2.imread(rec.image_path, cv2.IMREAD_GRAYSCALE)
        truth = (cv2.imread(rec.mask_path, cv2.IMREAD_UNCHANGED) >= 128).astype(np.uint8)
        if truth.ndim == 3:
            truth = truth[:, :, 0]
        pred = segmenter.predict_gray(gray).mask
        ax.imshow(gray, cmap="gray")
        if truth.any():
            ax.contour(truth, levels=[0.5], colors=[OLIVE], linewidths=1.4)
        if pred.any():
            ax.contour(pred, levels=[0.5], colors=[ACCENT], linewidths=1.4)
        ax.set_title(f"{row['source']} | Dice {row['dice']:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Test cases - worst row 1, median row 2, best row 3  (olive truth, coral prediction)")
    _save(fig, out_dir, "09_qualitative.png")


def _cuda(cfg: TrainConfig) -> bool:
    import torch

    return torch.cuda.is_available() and cfg.gpus > 0


# ─────────────────────────────────────────────────────────────────────────────
# Text report
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}"


def write_report(
    cfg: TrainConfig,
    out_dir: str,
    *,
    env: dict[str, Any],
    audits: dict[str, SourceAudit],
    cross_source: dict[str, Any],
    split_log: list[dict[str, Any]],
    split_report: dict[str, Any],
    cache_log: list[dict[str, Any]],
    spec_text: str,
    summary: Optional[dict[str, Any]],
) -> str:
    history = _load_json(os.path.join(out_dir, "metrics", "history.json"), [])
    done = _load_json(os.path.join(out_dir, "metrics", "training_done.json"), {})
    lines: list[str] = []
    add = lines.append
    rule = "=" * 88

    add(rule)
    add("OPENMED — MEDNEXT BRAIN-LESION SEGMENTATION — TRAINING REPORT")
    add(rule)
    add(f"generated   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"model       : {spec_text}")
    add("init        : random (no pretrained weights exist for 2D MedNeXt)")
    add(f"host        : {env.get('platform')} | torch {env.get('torch')} | cuda {env.get('cuda')} | "
        f"{env.get('gpu_count')} x {env.get('gpu_name')}")
    add("")
    add("1. DATASETS AND HYGIENE")
    add("-" * 88)
    for name, audit in audits.items():
        add(f"[{name}] {audit.summary['n_paired']} pairs -> kept {audit.summary['n_kept']}, "
            f"dropped {audit.summary['n_dropped']} {audit.summary['drop_reasons']}")
        for check in audit.checks.to_list():
            add(f"   {check['code']:<7}{check['name']:<26}{check['status']:<5}{check['detail']}")
    add(f"[cross-source] verdict {cross_source.get('verdict')} | pixel-confirmed shared images "
        f"{cross_source.get('confirmed_shared_images')} | exact pixel overlap "
        f"{cross_source.get('exact_pixels_shared')}")
    add("")
    add("2. SPLIT")
    add("-" * 88)
    for check in split_log + cache_log:
        add(f"   {check['code']:<7}{check['name']:<26}{check['status']:<5}{check['detail']}")
    add(f"   group leaks: {split_report.get('n_group_leaks')} | {split_report.get('caveat')}")
    add("")
    add("3. TRAINING")
    add("-" * 88)
    add(f"epochs run {done.get('epochs_run', len(history))} of {cfg.epochs} | early stopped "
        f"{done.get('stopped_early')} (patience {cfg.patience}) | wall time "
        f"{_fmt(done.get('elapsed_hours'), 2)} h")
    add(f"optimiser AdamW lr {cfg.lr} wd {cfg.weight_decay} | warmup {cfg.warmup_epochs} ep | cosine to "
        f"{cfg.min_lr_ratio:.0%} | batch {cfg.batch_size} x {cfg.gpus} GPU x accum {cfg.grad_accum} | "
        f"EMA {cfg.ema_decay} | drop-path {cfg.drop_path} | deep supervision {cfg.deep_supervision}")
    add(f"loss {cfg.bce_weight} BCE + {cfg.dice_weight} soft Dice | augmentation {cfg.aug_enabled}")
    if history:
        best = max(history, key=lambda h: h["best_score"])
        add(f"best validation macro Dice {_fmt(done.get('best', {}).get('score'))} at epoch "
            f"{done.get('best', {}).get('epoch')} ({done.get('best', {}).get('source')} weights)")
        add(f"final epoch train loss {_fmt(history[-1]['train_loss'])} | val EMA Dice "
            f"{_fmt(history[-1]['val_macro_ema'])} | skipped non-finite steps "
            f"{sum(h['skipped_steps'] for h in history)} | best-row {best['epoch']}")
    add("")
    if summary:
        tune = summary["tuning"]["chosen"]
        add("4. POST-PROCESSING (searched on validation only)")
        add("-" * 88)
        add(f"threshold {tune['threshold']:.2f} | min component area {tune['min_area_frac']:.4f} of frame | "
            f"flip-TTA {tune['tta_hflip']} | val macro Dice {_fmt(tune['val_macro_dice_default'])} "
            f"-> {_fmt(tune['val_macro_dice_tuned'])}")
        add("")
        add("5. HELD-OUT TEST (evaluated once, through the serving pipeline, original resolution)")
        add("-" * 88)
        test = summary["test"]
        if test:
            o = test["overall"]
            add(f"images {o['dice']['n']} | Dice mean {_fmt(o['dice']['mean'])} median "
                f"{_fmt(o['dice']['median'])} (p5 {_fmt(o['dice']['p5'])}) | IoU {_fmt(o['iou']['mean'])} | "
                f"precision {_fmt(o['precision']['mean'])} | recall {_fmt(o['recall']['mean'])}")
            add(f"HD95 median {_fmt(o['hd95_px']['median'], 1)} px (undefined for {test['hd95_undefined']}) | "
                f"missed lesions {test['missed_lesions']} | ECE {_fmt(test['calibration']['ece'])} | "
                f"median inference {_fmt(test['inference_ms_median'], 0)} ms")
            for name, entry in test["by_source"].items():
                add(f"   {name:<8} n={entry['dice']['n']:<5} Dice {_fmt(entry['dice']['mean'])} "
                    f"(median {_fmt(entry['dice']['median'])}) IoU {_fmt(entry['iou']['mean'])} "
                    f"HD95 {_fmt(entry['hd95_px']['median'], 1)} px")
            add("   by lesion size:")
            for label, entry in test["by_size"].items():
                add(f"      {label:<20} n={entry['n']:<5} Dice {_fmt(entry['dice']['mean'])} "
                    f"recall {_fmt(entry['recall']['mean'])}")
            add(f"   serve-vs-canvas |dDice| {_fmt(test['serve_vs_canvas']['mean_abs_dice_difference'])}")
            if summary.get("test_no_tta"):
                add(f"   without TTA: Dice {_fmt(summary['test_no_tta']['overall']['dice']['mean'])}")
            add("   worst cases: " + ", ".join(
                f"{w['stem']} ({w['dice']:.2f})" for w in test["worst"][:5]))
    add("")
    add("6. LIMITATIONS — read before quoting a number")
    add("-" * 88)
    for text in (
        "OpenMed group keys are filename-derived lesion/series keys, not verified patient ids.",
        "BTSC has no patient id; its split uses similarity-snapped contiguous blocks. Residual "
        "scan straddling is quantified in check V5 and cannot be proven zero.",
        "Both sources are public archives. Held-out slices of the same archive are not an external "
        "cohort; expect a drop on a different scanner or protocol.",
        "The model is binary (lesion vs background) and was trained only on slices that contain a "
        "lesion. It is meant to run after the classifier says 'not Normal'; on a lesion-free slice it "
        "has never been shown a negative and may still return a region.",
        "HD95 is in pixels; no pixel spacing is validated for these exports.",
        "Metrics use ground-truth masks from the source archives; their annotation protocol is unverified.",
    ):
        add(f"  - {text}")
    add("")
    add("7. FILES")
    add("-" * 88)
    add(f"  best/{CHECKPOINT_NAME}      <- the deployable model; copy to mednext/checkpoints/")
    add("  last.pt                  resumable state (optimizer, EMA, scaler)")
    add("  metrics/                 epochs.csv, history.json, summary.json, test_per_image.json, ...")
    add("  plots/                   01-09 figures")
    add("  splits/                  train/val/test manifests, purged_eval_slices.csv")
    add("  logs/                    run.log, rank*.log, dropped_*.csv, hygiene.json, TRAINING_REPORT.txt")
    add(rule)

    text = "\n".join(lines) + "\n"
    path = os.path.join(out_dir, "logs", "TRAINING_REPORT.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_zip(out_dir: str, zip_dir: str) -> str:
    os.makedirs(zip_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(zip_dir, f"mednext_brain_seg_{stamp}.zip")
    base = os.path.basename(out_dir.rstrip("/"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for root, _, files in os.walk(out_dir):
            for name in sorted(files):
                if name.endswith(".tmp"):
                    continue
                full = os.path.join(root, name)
                archive.write(full, os.path.join(base, os.path.relpath(full, out_dir)))
    best = os.path.join(out_dir, "best", CHECKPOINT_NAME)
    if os.path.isfile(best):
        import shutil

        shutil.copy2(best, os.path.join(zip_dir, CHECKPOINT_NAME))
    logger.info("ZIP ready: %s (%.1f MB)", zip_path, os.path.getsize(zip_path) / 2**20)
    return zip_path


def write_epochs_csv_check(out_dir: str) -> int:
    path = os.path.join(out_dir, "metrics", "epochs.csv")
    if not os.path.isfile(path):
        return 0
    with open(path, newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def environment_snapshot() -> dict[str, Any]:
    import torch

    cuda = torch.cuda.is_available()
    return {
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda if cuda else None,
        "gpu_count": torch.cuda.device_count() if cuda else 0,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else "cpu",
    }
