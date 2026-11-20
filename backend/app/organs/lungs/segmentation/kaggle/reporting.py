"""Plots, the plain-text training report, and the Kaggle export ZIP.

Every figure is written to ``<run>/plots`` and numbered in the brain module's style.
Plotting never raises into the run: a failed figure is logged and skipped, because
losing a plot must not cost the trained model its export.

Every number in ``TRAINING_REPORT.txt`` is read from a file the run wrote
(``hygiene.json``, ``split_report.json``, ``history.json``, ``metrics/summary.json``,
``training_done.json``). Nothing is typed in from expectation.
"""

from __future__ import annotations

import json
import logging
import os
import time
import zipfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from ..checkpoint import load_serving_checkpoint
from ..labels import CLASS_COVID, CLASS_NON_COVID, CLASS_NORMAL
from ..pipeline import LungSegmenter
from ..postprocess import LungPost, OpacityPost
from .augment import augment
from .config import TrainConfig
from .data import ArrayCache
from .splits import SplitResult

logger = logging.getLogger("reporting")

PAPER, INK, CORAL, OLIVE, MUSTARD, MUTE = (
    "#f7f1de", "#15140f", "#ed6f5c", "#6b7a3a", "#c9a227", "#8b8676",
)
LUNG_BLUE = "#4a7a9c"          # the lung-field wash used by the frontend panel
STAGE_COLOURS = {"lung": OLIVE, "twohead": CORAL}
CLASS_COLOURS = {CLASS_COVID: CORAL, CLASS_NON_COVID: MUSTARD, CLASS_NORMAL: LUNG_BLUE}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
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


def _load_json(path: str, default: Any = None) -> Any:
    if not os.path.isfile(path):
        return default
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not np.isfinite(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


# ─────────────────────────────────────────────────────────────────────────────
# 01 — learning curves
# ─────────────────────────────────────────────────────────────────────────────


@_safely
def plot_learning_curves(stages: dict[str, list[dict[str, Any]]], out_dir: str) -> None:
    stages = {name: rows for name, rows in stages.items() if rows}
    if not stages:
        return
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    final = "twohead" if "twohead" in stages else next(iter(stages))

    def series(name: str, key: str) -> tuple[list[int], list[float]]:
        rows = stages.get(name, [])
        epochs = [int(row["epoch"]) for row in rows if row.get(key) is not None]
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return epochs, values

    ax = axes[0, 0]
    for name in stages:
        epochs, train = series(name, "train_loss")
        ax.plot(epochs, train, color=STAGE_COLOURS.get(name, INK), lw=1.4,
                label=f"{name} train")
        epochs, val = series(name, "val_loss_ema")
        ax.plot(epochs, val, color=STAGE_COLOURS.get(name, INK), lw=1, ls="--",
                label=f"{name} val EMA")
    ax.set(title="Loss", xlabel="epoch")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for name in stages:
        epochs, raw = series(name, "val_lung_dice_raw")
        ax.plot(epochs, raw, color=STAGE_COLOURS.get(name, INK), lw=1,
                alpha=0.55, label=f"{name} raw")
        epochs, ema = series(name, "val_lung_dice_ema")
        ax.plot(epochs, ema, color=STAGE_COLOURS.get(name, INK), lw=1.8,
                label=f"{name} EMA")
    ax.set(title="Lung-field Dice (validation)", xlabel="epoch", ylim=(0, 1))
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    epochs, dice = series(final, "val_opacity_pos_dice_ema")
    if dice:
        ax.plot(epochs, dice, color=CORAL, lw=1.8, label="opacity positive Dice (EMA)")
    epochs, fpr = series(final, "val_normal_fpr_ema")
    if fpr:
        ax.plot(epochs, fpr, color=INK, lw=1.2, label="Normal false-positive rate")
    ax.set(title="Head B (validation)", xlabel="epoch", ylim=(0, 1))
    if dice or fpr:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "head B not trained in this run", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)

    ax = axes[1, 0]
    for name in stages:
        epochs, score = series(name, "val_score_ema")
        ax.plot(epochs, score, color=STAGE_COLOURS.get(name, INK), lw=1.4, label=name)
    ax.set(title="Selection score (EMA)", xlabel="epoch")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    for name in stages:
        epochs, lr = series(name, "lr")
        ax.plot(epochs, lr, color=STAGE_COLOURS.get(name, INK), lw=1.2, label=name)
    ax.set(title="Learning rate (warm-up + cosine)", xlabel="epoch", yscale="log")
    ax.legend(fontsize=7)

    ax = axes[1, 2]
    for name in stages:
        epochs, seconds = series(name, "epoch_seconds")
        ax.plot(epochs, np.array(seconds) / 60.0, color=STAGE_COLOURS.get(name, INK),
                lw=1.2, label=name)
    ax.set(title="Epoch wall time", xlabel="epoch", ylabel="minutes")
    ax.legend(fontsize=7)

    _save(fig, out_dir, "01_learning_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# 02 — dataset overview
# ─────────────────────────────────────────────────────────────────────────────


@_safely
def plot_dataset_overview(
    split_report: dict[str, Any], hygiene: dict[str, Any], out_dir: str
) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    counts = split_report.get("counts_after") or {}

    ax = axes[0]
    splits = ["train", "val", "test"]
    classes = [CLASS_COVID, CLASS_NON_COVID, CLASS_NORMAL]
    bottom = np.zeros(len(splits))
    for name in classes:
        values = np.array([counts.get(split, {}).get(name, 0) for split in splits])
        ax.bar(splits, values, bottom=bottom, color=CLASS_COLOURS[name], label=name,
               edgecolor=INK, linewidth=0.4)
        bottom += values
    for index, total in enumerate(bottom):
        ax.text(index, total, f"{int(total):,}", ha="center", va="bottom", fontsize=9)
    ax.set(title="Class x split after re-grouping", ylabel="images")
    ax.legend(fontsize=8)

    ax = axes[1]
    area = next((c["data"]["area_px"] for c in hygiene.get("checks", []) if c["id"] == "H06"), {})
    if area:
        keys = ["min", "p1", "median", "p99", "max"]
        values = [area.get(key, 0) for key in keys]
        ax.bar(keys, values, color=LUNG_BLUE, edgecolor=INK, linewidth=0.4)
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=8)
        ax.set(title="Lung-mask area (px)", ylabel="pixels")
    else:
        ax.text(0.5, 0.5, "no H06 data", ha="center", transform=ax.transAxes)

    ax = axes[2]
    morphology = next(
        (c["data"] for c in hygiene.get("checks", []) if c["id"] == "H07"), {}
    )
    histogram = morphology.get("components_hist") or {}
    if histogram:
        labels = [f"{key} comp" for key in sorted(histogram, key=int)]
        values = [histogram[key] for key in sorted(histogram, key=int)]
        ax.bar(labels, values, color=OLIVE, edgecolor=INK, linewidth=0.4)
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
        ax.set(title="Lung-mask components", ylabel="images")
        ax.set_ylim(0, max(values) * 1.15)
    else:
        ax.text(0.5, 0.5, "no H07 data", ha="center", transform=ax.transAxes)

    _save(fig, out_dir, "02_dataset_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
# 03 — duplicate clusters and the move against the shipped split
# ─────────────────────────────────────────────────────────────────────────────


@_safely
def plot_duplicate_clusters(split_report: dict[str, Any], out_dir: str) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    distribution = split_report.get("cluster_size_distribution_surviving") or {}
    if distribution:
        keys = sorted(distribution, key=int)
        values = [distribution[key] for key in keys]
        ax.bar([f"{key}" for key in keys], values, color=OLIVE, edgecolor=INK, linewidth=0.4)
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
        ax.set_yscale("log")
        ax.set(title="Duplicate cluster sizes (surviving)", xlabel="images per cluster",
               ylabel="clusters (log)")

    ax = axes[1]
    before = split_report.get("cross_split_before") or {}
    after = split_report.get("cross_split_after") or {}
    metrics = ["exact_pixel_pairs", "phash_confirmed_pairs", "clusters_spanning_splits"]
    width = 0.36
    positions = np.arange(len(metrics))
    ax.bar(positions - width / 2, [before.get(key, 0) for key in metrics], width,
           color=CORAL, label="shipped split", edgecolor=INK, linewidth=0.4)
    ax.bar(positions + width / 2, [after.get(key, 0) for key in metrics], width,
           color=OLIVE, label="this split", edgecolor=INK, linewidth=0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels([key.replace("_", "\n") for key in metrics], fontsize=7)
    ax.set(title="Cross-split duplicates: before vs after", ylabel="pairs / clusters")
    ax.legend(fontsize=8)

    ax = axes[2]
    matrix = split_report.get("moved_matrix") or {}
    splits = ["train", "val", "test"]
    grid = np.zeros((len(splits), len(splits)), dtype=np.float64)
    # keys are serialised as "shipped->new" so the JSON report stays flat
    for key, count in matrix.items():
        source, _, target = str(key).partition("->")
        if source in splits and target in splits:
            grid[splits.index(source), splits.index(target)] = count
    image = ax.imshow(grid, cmap="YlOrBr", aspect="auto")
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels(splits)
    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels(splits)
    ax.set(xlabel="new split", ylabel="shipped split",
           title="Images moved against the shipped split")
    for row in range(len(splits)):
        for column in range(len(splits)):
            value = int(grid[row, column])
            if value:
                ax.text(column, row, f"{value:,}", ha="center", va="center", fontsize=8,
                        color=INK if value < grid.max() * 0.6 else PAPER)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    _save(fig, out_dir, "03_duplicate_clusters.png")


# ─────────────────────────────────────────────────────────────────────────────
# 04 — training samples, 05 — augmentation preview
# ─────────────────────────────────────────────────────────────────────────────


def _sample_records(split: SplitResult, per_class: int, seed: int) -> list[Any]:
    rng = np.random.default_rng(seed)
    chosen = []
    for name in (CLASS_COVID, CLASS_NON_COVID, CLASS_NORMAL):
        pool = [record for record in split.part("train") if record.cls == name]
        if not pool:
            continue
        index = rng.choice(len(pool), size=min(per_class, len(pool)), replace=False)
        chosen.extend(pool[int(i)] for i in sorted(index))
    return chosen


@_safely
def plot_samples(cache: ArrayCache, split: SplitResult, cfg: TrainConfig, out_dir: str) -> None:
    records = _sample_records(split, 3, cfg.seed)
    if not records:
        return
    _style()
    fig, axes = plt.subplots(3, 3, figsize=(10, 10.5))
    for ax, record in zip(axes.ravel(), records, strict=False):
        image = cache.image(record.uid)
        lung = cache.lung(record.uid)
        opacity = cache.infection(record.uid)
        shown = np.dstack([image] * 3).astype(np.float32) / 255.0
        if opacity is not None and opacity.any():
            overlay = shown.copy()
            overlay[opacity > 0] = np.array([237, 111, 92]) / 255.0
            shown = 0.55 * shown + 0.45 * overlay
        ax.imshow(shown)
        if lung.any():
            ax.contour(lung, levels=[0.5], colors=[LUNG_BLUE], linewidths=1.2)
        ax.set_title(f"{record.cls}\n{record.filename}", fontsize=8)
        ax.axis("off")
    handles = [
        Patch(color=LUNG_BLUE, label="lung field (contour)"),
        Patch(color=CORAL, label="COVID opacity (GT, filled)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    _save(fig, out_dir, "04_training_samples.png")


@_safely
def plot_augmentation_preview(
    cache: ArrayCache, split: SplitResult, cfg: TrainConfig, out_dir: str
) -> None:
    pool = [record for record in split.part("train") if record.opacity_supervised]
    if not pool:
        return
    record = pool[0]
    image = cache.image(record.uid)
    lung = cache.lung(record.uid)
    opacity = cache.infection(record.uid)

    _style()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for step, ax in enumerate(axes.ravel()):
        if step == 0:
            shown, shown_lung, shown_opacity = image, lung, opacity
            title = "original"
        else:
            rng = np.random.default_rng([cfg.seed, step])
            shown, shown_lung, shown_opacity = augment(image, lung, opacity, rng, cfg)
            title = f"augmented #{step}"
        canvas = np.dstack([shown] * 3).astype(np.float32) / 255.0
        if shown_opacity is not None and shown_opacity.any():
            overlay = canvas.copy()
            overlay[shown_opacity > 0] = np.array([237, 111, 92]) / 255.0
            canvas = 0.55 * canvas + 0.45 * overlay
        ax.imshow(canvas)
        if shown_lung.any():
            ax.contour(shown_lung, levels=[0.5], colors=[LUNG_BLUE], linewidths=1.0)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    _save(fig, out_dir, "05_augmentation_preview.png")


# ─────────────────────────────────────────────────────────────────────────────
# 06 — test distributions
# ─────────────────────────────────────────────────────────────────────────────


@_safely
def plot_test_distributions(
    summary: dict[str, Any], rows: list[dict[str, Any]], out_dir: str
) -> None:
    if not rows:
        return
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    dice = np.array([row["lung_dice"] for row in rows], dtype=np.float64)
    ax.hist(dice, bins=40, color=LUNG_BLUE, edgecolor=INK, linewidth=0.3)
    mean = summary["lung_field"]["dice"]["mean"]
    median = summary["lung_field"]["dice"]["median"]
    ax.axvline(mean, color=CORAL, ls="--", lw=1.4, label=f"mean {mean:.3f}")
    ax.axvline(median, color=INK, ls=":", lw=1.4, label=f"median {median:.3f}")
    ax.set(title="Lung-field Dice per image (test)", xlabel="Dice", ylabel="images")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    labels, data = [], []
    for name in (CLASS_COVID, CLASS_NON_COVID, CLASS_NORMAL):
        values = [row["lung_dice"] for row in rows if row["class"] == name]
        if values:
            labels.append(name)
            data.append(values)
    if data:
        box = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
        for patch, name in zip(box["boxes"], labels, strict=False):
            patch.set_facecolor(CLASS_COLOURS.get(name, MUTE))
            patch.set_alpha(0.75)
        for median in box["medians"]:
            median.set_color(INK)
    ax.set(title="Lung-field Dice by class", ylabel="Dice")

    ax = axes[1, 0]
    opacity = summary.get("opacity")
    if opacity:
        bins = list(opacity["positives"]["by_size"].items())
        names = [name.split(" ")[0] for name, _ in bins]
        means = [(block["dice"]["mean"] if block["dice"]["mean"] is not None else 0.0)
                 for _, block in bins]
        counts = [block["n"] for _, block in bins]
        positions = np.arange(len(bins))
        ax.bar(positions, means, color=CORAL, edgecolor=INK, linewidth=0.4)
        for index, (value, count) in enumerate(zip(means, counts, strict=True)):
            ax.text(index, value, f"{value:.3f}\nn={count}", ha="center", va="bottom",
                    fontsize=8)
        ax.set_xticks(positions)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1.05)
        ax.set(title="Opacity Dice on COVID positives, by lesion size", ylabel="Dice")
    else:
        ax.text(0.5, 0.5, "no opacity head", ha="center", transform=ax.transAxes)

    ax = axes[1, 1]
    if opacity:
        normal = [row for row in rows if row["class"] == CLASS_NORMAL and row["opacity_evaluated"]]
        fractions = [row["opacity_pred_fraction_of_lung"] or 0.0 for row in normal]
        ax.hist(fractions, bins=40, color=OLIVE, edgecolor=INK, linewidth=0.3)
        rate = opacity["normal"]["false_positive_rate"]
        ax.set(title=f"Normal predicted opacity fraction (test)\n"
                     f"{opacity['normal']['n_with_prediction']} / {opacity['normal']['n']} "
                     f"images with any pixel = {_fmt(rate)}",
               xlabel="predicted opacity / lung area", ylabel="images")
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "no opacity head", ha="center", transform=ax.transAxes)

    _save(fig, out_dir, "06_test_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 07 — post-processing search
# ─────────────────────────────────────────────────────────────────────────────


def _heatmap(ax: plt.Axes, block: dict[str, float], title: str) -> None:
    thresholds = sorted({key.split("|")[0] for key in block}, key=float)
    fractions = sorted({key.split("|")[1] for key in block}, key=float)
    grid = np.array([[block[f"{threshold}|{fraction}"] for fraction in fractions]
                     for threshold in thresholds], dtype=np.float64)
    ax.imshow(grid, cmap="YlOrBr", aspect="auto")
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f"{float(value):.4f}" for value in fractions], fontsize=7, rotation=45)
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels(thresholds, fontsize=7)
    ax.set(xlabel="minimum area fraction", ylabel="threshold", title=title)
    best = np.unravel_index(int(np.nanargmax(grid)), grid.shape)
    ax.scatter([best[1]], [best[0]], s=60, facecolors="none", edgecolors=INK, linewidths=1.6)
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            ax.text(column, row, f"{grid[row, column]:.3f}", ha="center", va="center",
                    fontsize=5.5, color=INK)


@_safely
def plot_postprocess_grid(grids: dict[str, Any], out_dir: str) -> None:
    lung = grids.get("lung") or {}
    opacity = grids.get("opacity") or {}
    keys = [(name, block) for name, block in lung.items() if block]
    if not keys and not opacity:
        return
    _style()
    rows = 2
    fig, axes = plt.subplots(rows, 2, figsize=(13, 9))
    for index, (name, block) in enumerate(keys[:2]):
        _heatmap(axes[0, index], block, f"Lung Dice ({name})")
    for index in range(len(keys), 2):
        axes[0, index].axis("off")
    opacity_keys = [(name, block) for name, block in opacity.items() if block]
    for index, (name, block) in enumerate(opacity_keys[:2]):
        _heatmap(axes[1, index], block, f"Opacity selection score ({name})")
    for index in range(len(opacity_keys), 2):
        axes[1, index].axis("off")
        axes[1, index].text(0.5, 0.5, "no opacity head", ha="center",
                            transform=axes[1, index].transAxes)
    _save(fig, out_dir, "07_postprocess_search.png")


# ─────────────────────────────────────────────────────────────────────────────
# 08 — calibration and precision-recall for head B
# ─────────────────────────────────────────────────────────────────────────────


@_safely
def plot_calibration(summary: dict[str, Any], out_dir: str) -> None:
    opacity = summary.get("opacity")
    if not opacity:
        return
    calibration = opacity.get("calibration")
    pixels_path = os.path.join(out_dir, "metrics", "test_calibration_pixels.npz")
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    if calibration:
        confidence = np.array(calibration["bin_confidence"])
        accuracy = np.array(calibration["bin_accuracy"])
        weight = np.array(calibration["bin_weight"])
        mask = weight > 0
        ax.plot([0, 1], [0, 1], color=MUTE, ls="--", lw=1)
        ax.plot(confidence[mask], accuracy[mask], color=CORAL, marker="o", lw=1.5,
                label=f"ECE {calibration['ece']:.4f}")
        sizes = 30 + 400 * weight[mask]
        ax.scatter(confidence[mask], accuracy[mask], s=sizes, facecolors="none",
                   edgecolors=INK, linewidths=0.8)
        ax.set(title=f"Reliability, opacity head\n{calibration['pixels']:,} sampled pixels "
                     "inside the predicted lung field",
               xlabel="predicted probability", ylabel="observed frequency", xlim=(0, 1),
               ylim=(0, 1))
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no calibration data", ha="center", transform=ax.transAxes)

    ax = axes[1]
    if os.path.isfile(pixels_path):
        with np.load(pixels_path) as data:
            probs = data["probs"].astype(np.float32)
            truth = data["truth"].astype(bool)
        thresholds = np.linspace(0.05, 0.95, 91)
        precision, recall = [], []
        for threshold in thresholds:
            predicted = probs >= threshold
            tp = int(np.count_nonzero(predicted & truth))
            fp = int(np.count_nonzero(predicted & ~truth))
            fn = int(np.count_nonzero(~predicted & truth))
            precision.append(tp / max(tp + fp, 1))
            recall.append(tp / max(tp + fn, 1))
        ax.plot(recall, precision, color=CORAL, lw=1.8)
        ax.set(title="Pixel precision-recall, opacity head (test)", xlabel="recall",
               ylabel="precision", xlim=(0, 1), ylim=(0, 1))
        positive_rate = float(truth.mean())
        ax.axhline(positive_rate, color=MUTE, ls=":", lw=1,
                   label=f"prevalence {positive_rate:.4f}")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no calibration pixels", ha="center", transform=ax.transAxes)

    _save(fig, out_dir, "08_calibration_pr.png")


# ─────────────────────────────────────────────────────────────────────────────
# 09 — qualitative grid
# ─────────────────────────────────────────────────────────────────────────────


@_safely
def plot_qualitative(
    cache: ArrayCache,
    split: SplitResult,
    rows: list[dict[str, Any]],
    out_dir: str,
    checkpoint_path: str,
) -> None:
    """Best / median / worst test cases, re-run through the shipped checkpoint.

    The masks drawn are what the application produces: the image is decoded from its
    bytes, segmented by :class:`LungSegmenter` with the tuned post-processing stored in
    the checkpoint, then overlaid. The prediction is not carried over from evaluation,
    so a serving regression would show up here.
    """
    from .metrics import SIZE_BINS

    positive = [row for row in rows if row["opacity_dice"] is not None]
    if positive:
        def bin_of(row: dict[str, Any]) -> int:
            fraction = row["opacity_gt_fraction_of_lung"]
            for index, (_, low, high) in enumerate(SIZE_BINS):
                if low <= fraction < high:
                    return index
            return len(SIZE_BINS) - 1

        panels = []
        for index, (label, _, _) in enumerate(SIZE_BINS):
            members = [row for row in positive if bin_of(row) == index]
            members.sort(key=lambda row: row["opacity_dice"])
            if not members:
                continue
            picks = [members[0], members[len(members) // 2], members[-1]]
            panels.append((label, picks))
        title_suffix = "opacity Dice by lesion size"
    else:
        ordered = sorted(rows, key=lambda row: row["lung_dice"])
        picks = [ordered[0], ordered[len(ordered) // 2], ordered[-1]] if ordered else []
        panels = [("lung field", picks)]
        title_suffix = "lung Dice"

    if not panels:
        return

    payload = load_serving_checkpoint(checkpoint_path)
    postprocess = payload.get("postprocess") or {}
    lung_post = LungPost.from_dict(postprocess.get("lung") or {})
    opacity_post = (OpacityPost.from_dict(postprocess["opacity"])
                    if postprocess.get("opacity") else None)
    segmenter = LungSegmenter(checkpoint_path)
    try:
        by_name = {record.filename: record for record in split.part("test")}
        _style()
        fig, axes = plt.subplots(len(panels), 3, figsize=(10, 3.4 * len(panels)),
                                 squeeze=False)
        for row_index, (label, picks) in enumerate(panels):
            for column in range(3):
                ax = axes[row_index, column]
                if column >= len(picks):
                    ax.axis("off")
                    continue
                row = picks[column]
                record = by_name.get(row["filename"])
                if record is None:
                    ax.axis("off")
                    continue
                with open(record.image_path, "rb") as handle:
                    result = segmenter.predict_bytes(
                        handle.read(), lung_post=lung_post, opacity_post=opacity_post
                    )[0]
                image = cache.image(record.uid)
                truth = cache.infection(record.uid)
                shown = np.dstack([image] * 3).astype(np.float32) / 255.0
                if result.opacity_mask is not None and result.opacity_mask.any():
                    overlay = shown.copy()
                    overlay[result.opacity_mask > 0] = np.array([237, 111, 92]) / 255.0
                    shown = 0.55 * shown + 0.45 * overlay
                ax.imshow(shown)
                if result.lung_mask.any():
                    ax.contour(result.lung_mask, levels=[0.5], colors=[LUNG_BLUE],
                               linewidths=1.1)
                if truth is not None and truth.any():
                    ax.contour(truth, levels=[0.5], colors=[MUSTARD], linewidths=1.3,
                               linestyles="--")
                ax.set_title(
                    f"{label}\n{row['filename']}  "
                    + (f"opacity Dice {row['opacity_dice']:.3f}"
                       if row["opacity_dice"] is not None
                       else f"lung Dice {row['lung_dice']:.3f}"),
                    fontsize=8,
                )
                ax.axis("off")
    finally:
        segmenter.unload()

    handles = [
        Patch(color=LUNG_BLUE, label="predicted lung field (contour)"),
        Patch(color=CORAL, label="predicted opacity (filled)"),
        Patch(color=MUSTARD, label="ground-truth opacity (dashed contour)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"Best / median / worst by {title_suffix}", fontsize=10)
    _save(fig, out_dir, "09_qualitative.png")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────


def _table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(len(header))]
    line = "  ".join(str(value).ljust(widths[i]) for i, value in enumerate(header))
    body = "\n".join("  ".join(str(value).ljust(widths[i]) for i, value in enumerate(row))
                     for row in rows)
    return line + "\n" + "-" * len(line) + "\n" + body


def write_report(
    cfg: TrainConfig,
    out_dir: str,
    *,
    env: dict[str, Any] | None,
    hygiene: dict[str, Any],
    split_report: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    summary: dict[str, Any] | None,
    history: dict[str, list[dict[str, Any]]],
) -> str:
    """Write ``TRAINING_REPORT.txt`` from the files this run produced."""
    test = (summary or {}).get("test") or {}
    lung = test.get("lung_field") or {}
    opacity = test.get("opacity")
    lines: list[str] = []
    lines.append("OpenMed — lungs segmentation (COVID opacity)")
    lines.append("=" * 60)
    lines.append(f"run directory : {out_dir}")
    lines.append(f"written       : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"stage trained : {', '.join(stages) or 'none'}")
    if env:
        lines.append(f"platform      : {env.get('platform')} | torch {env.get('torch')} | "
                     f"cuda {env.get('cuda')} | GPUs {env.get('gpu_count')} x "
                     f"{env.get('gpu_name')}")
    lines.append("")
    lines.append("DATA")
    lines.append("-" * 60)
    counts = hygiene.get("counts", {})
    lines.append(_table(
        [["images catalogued", f"{counts.get('records_catalogued', 0):,}"],
         ["images kept", f"{counts.get('records_kept', 0):,}"],
         ["images dropped", f"{counts.get('records_dropped', 0):,}"]],
        ["hygiene", "value"],
    ))
    tally = hygiene.get("tally", {})
    lines.append("")
    lines.append("checks: " + ", ".join(f"{name} {count}" for name, count in tally.items()))
    for check in hygiene.get("checks", []):
        if check["status"] in ("FAIL", "WARN"):
            lines.append(f"  {check['id']} {check['status']:<4} {check['name']}: {check['detail']}")
    lines.append("")
    before = split_report.get("counts_before_shipped") or {}
    after = split_report.get("counts_after") or {}
    lines.append("SPLIT (before -> after, images)")
    lines.append("-" * 60)
    rows = []
    for split in ("train", "val", "test"):
        for name in (CLASS_COVID, CLASS_NON_COVID, CLASS_NORMAL):
            rows.append([split, name, f"{before.get(split, {}).get(name, 0):,}",
                         f"{after.get(split, {}).get(name, 0):,}"])
    lines.append(_table(rows, ["split", "class", "shipped", "this split"]))
    lines.append("")
    lines.append(
        f"images moved against the shipped split: "
        f"{split_report.get('images_moved', 0):,} "
        f"({split_report.get('images_moved_fraction', 0.0):.1%})"
    )
    cross = split_report.get("cross_split_before") or {}
    solved = split_report.get("cross_split_after") or {}
    lines.append("")
    lines.append(_table(
        [[key, str(cross.get(key, 0)), str(solved.get(key, 0))]
         for key in sorted(set(cross) | set(solved))],
        ["cross-split duplicates", "shipped", "this split"],
    ))
    lines.append("")
    lines.append("TRAINING")
    lines.append("-" * 60)
    for name, done in stages.items():
        history_rows = history.get(name, [])
        best = max((row["best_score"] for row in history_rows), default=None)
        best_epoch = next(
            (row["epoch"] for row in history_rows if row["best_score"] == best), None
        )
        lines.append(
            f"{name:<9} epochs {len(history_rows):>3} | best selection score "
            f"{_fmt(best)} at epoch {best_epoch} | stopped "
            f"{'early' if done.get('stopped_early') else 'by budget' if done.get('stopped_by_time_budget') else 'at schedule'} "
            f"| {done.get('elapsed_hours', 0.0):.2f} h"
        )
    lines.append("")
    postprocess = (summary or {}).get("postprocess") or {}
    lines.append("POST-PROCESSING (chosen on validation)")
    lines.append("-" * 60)
    lung_post = postprocess.get("lung") or {}
    opacity_post = postprocess.get("opacity") or {}
    lines.append(
        "lung    : threshold {:.2f}, min area {:.4f}, keep largest {}, fill holes {}, "
        "TTA {}".format(
            lung_post.get("threshold", 0.5), lung_post.get("min_area_frac", 0.0),
            lung_post.get("keep_largest_k", 2), lung_post.get("fill_holes", True),
            lung_post.get("tta_hflip", False),
        )
    )
    if opacity_post:
        lines.append(
            "opacity : threshold {:.2f}, min area {:.4f}, TTA {}".format(
                opacity_post.get("threshold", 0.5), opacity_post.get("min_area_frac", 0.0),
                opacity_post.get("tta_hflip", False),
            )
        )
    lines.append("")
    lines.append("HELD-OUT TEST (serving pipeline, original resolution)")
    lines.append("-" * 60)
    if lung:
        lines.append(
            f"lung field   Dice {_fmt(lung['dice']['mean'])} mean / "
            f"{_fmt(lung['dice']['median'])} median | IoU {_fmt(lung['iou']['mean'])} | "
            f"HD95 {_fmt(lung['hd95_px']['median'], 1)} px median | "
            f"{lung['dice']['n']:,} images"
        )
        for name, block in (lung.get("by_class") or {}).items():
            lines.append(f"  {name:<10} n={block['n']:>5}  Dice {_fmt(block['dice']['mean'])}")
    if opacity:
        positives = opacity["positives"]
        normal = opacity["normal"]
        non_covid = opacity["non_covid"]
        lines.append(
            f"opacity      positive Dice {_fmt(positives['dice']['mean'])} mean / "
            f"{_fmt(positives['dice']['median'])} median on {positives['n']:,} COVID-19 "
            f"images | HD95 {_fmt(positives['hd95_px']['median'], 1)} px median"
        )
        for name, block in positives["by_size"].items():
            lines.append(f"  {name:<24} n={block['n']:>4}  Dice {_fmt(block['dice']['mean'])}")
        lines.append(
            f"Normal       false-positive rate {_fmt(normal['false_positive_rate'])} "
            f"({normal['n_with_prediction']} / {normal['n']}) | mean predicted fraction "
            f"{_fmt(normal['mean_predicted_fraction_of_lung'])}"
        )
        lines.append(
            f"Non-COVID    any opacity {_fmt(non_covid['any_opacity_rate'])} | mean predicted "
            f"fraction {_fmt(non_covid['mean_predicted_fraction_of_lung'])}  (behaviour, not a score)"
        )
        calibration = opacity.get("calibration")
        if calibration:
            lines.append(f"calibration  ECE {_fmt(calibration['ece'])} on "
                         f"{calibration['pixels']:,} pixels inside the predicted lung field")
    lines.append("")
    lines.append("ARTEFACTS")
    lines.append("-" * 60)
    for folder, _, files in os.walk(out_dir):
        for name in sorted(files):
            if name.endswith((".tmp", ".pyc")):
                continue
            path = os.path.join(folder, name)
            relative = os.path.relpath(path, out_dir)
            lines.append(f"  {relative:<48} {os.path.getsize(path) / 2**20:8.2f} MB")
    lines.append("")
    lines.append("CAVEATS")
    lines.append("-" * 60)
    lines.append("  HD95 is in pixels; COVID-QU-Ex has no validated pixel spacing.")
    lines.append("  Opacity Dice is computed on COVID-19 positives only. Non-COVID is")
    lines.append("  reported as behaviour: its infection mask is an annotation negative")
    lines.append("  (bacterial consolidation was never delineated).")
    lines.append("  No SOTA claim is made: COVID-QU-Ex has no valid leaderboard and a")
    lines.append("  provenance confound across its four source classes.")
    path = os.path.join(out_dir, "TRAINING_REPORT.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    logger.info("report written: %s", path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# ZIP
# ─────────────────────────────────────────────────────────────────────────────


def export_zip(out_dir: str, zip_dir: str, *, name: str | None = None) -> str:
    """Zip the whole run directory. The ZIP alone must reproduce every claim."""
    os.makedirs(zip_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M")
    stem = name or f"lungs_seg_{os.path.basename(os.path.normpath(out_dir))}_{stamp}"
    path = os.path.join(zip_dir, f"{stem}.zip")
    count = 0
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for folder, directories, files in os.walk(out_dir):
            directories[:] = [d for d in directories if d != "__pycache__"]
            for file_name in sorted(files):
                if file_name.endswith((".tmp", ".pyc")):
                    continue
                full = os.path.join(folder, file_name)
                archive.write(full, os.path.relpath(full, os.path.dirname(out_dir)))
                count += 1
    logger.info("ZIP written: %s (%d files, %.1f MB)", path, count,
                os.path.getsize(path) / 2**20)
    return path
