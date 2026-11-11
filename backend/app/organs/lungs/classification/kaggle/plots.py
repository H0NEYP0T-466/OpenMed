"""Every figure the run produces. matplotlib only, Agg backend, dpi 140.

Rules for this module:
    * one function per figure, ``(data..., path) -> path``
    * nothing calls ``plt.show()`` and nothing opens a window
    * every function creates, saves and closes its own figure

The plotting code is kept away from the trainer so the trainer stays readable
and so a plot can be regenerated from a saved artefact without re-running
anything.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

DPI = 140
_COLOURS = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b")


def _save(figure: plt.Figure, path: str) -> str:
    target = os.path.abspath(path)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return target


def _empty(axis: plt.Axes, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=10, wrap=True)
    axis.axis("off")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level figures
# ─────────────────────────────────────────────────────────────────────────────


def plot_class_distribution(counts: Mapping[str, int], path: str) -> str:
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    names = list(counts)
    values = [int(counts[name]) for name in names]
    bars = axis.bar(range(len(names)), values, color=_COLOURS[: len(names)])
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=15, ha="right")
    axis.set_ylabel("images")
    axis.set_title(f"Class distribution (n={sum(values)})")
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    return _save(figure, path)


def plot_split_distribution(
    per_split: Mapping[str, Mapping[str, int]], class_names: Sequence[str], path: str
) -> str:
    splits = [name for name in ("train", "val", "test") if name in per_split]
    figure, axis = plt.subplots(figsize=(9.0, 4.6))
    width = 0.8 / max(len(splits), 1)
    positions = np.arange(len(class_names))
    for offset, split in enumerate(splits):
        values = [int(per_split[split].get(name, 0)) for name in class_names]
        axis.bar(
            positions + offset * width - 0.4 + width / 2,
            values,
            width=width,
            label=f"{split} (n={sum(values)})",
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(class_names, rotation=15, ha="right")
    axis.set_ylabel("images")
    axis.set_title("Split distribution per class (cluster-grouped)")
    axis.legend(fontsize=8)
    return _save(figure, path)


def plot_sample_grid_per_class(
    samples: Mapping[str, Sequence[np.ndarray]], path: str, per_class: int = 4
) -> str:
    class_names = list(samples)
    columns = max(1, per_class)
    figure, axes = plt.subplots(
        max(1, len(class_names)),
        columns,
        figsize=(2.6 * columns, 2.6 * max(1, len(class_names))),
        squeeze=False,
    )
    for row, name in enumerate(class_names):
        images = list(samples[name])[:columns]
        for column in range(columns):
            axis = axes[row][column]
            if column < len(images):
                image = np.asarray(images[column])
                axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
                if column == 0:
                    axis.set_ylabel(name, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle("Samples per class", fontsize=11)
    return _save(figure, path)


def plot_augmentation_preview(
    panels: Sequence[tuple[str, np.ndarray]], path: str, title: str = "Augmentation preview"
) -> str:
    count = max(1, len(panels))
    figure, axes = plt.subplots(1, count, figsize=(2.4 * count, 2.9), squeeze=False)
    for column in range(count):
        axis = axes[0][column]
        if column < len(panels):
            label, image = panels[column]
            image = np.asarray(image)
            axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
            axis.set_title(label, fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title, fontsize=11)
    return _save(figure, path)


def plot_duplicate_cluster_histogram(cluster_sizes: Sequence[int], path: str) -> str:
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    sizes = np.asarray(list(cluster_sizes), dtype=np.int64)
    if sizes.size == 0:
        _empty(axis, "no clusters")
        return _save(figure, path)
    maximum = int(sizes.max())
    bins = np.arange(1, maximum + 2) - 0.5
    axis.hist(sizes, bins=bins, color="#1f77b4", edgecolor="white")
    axis.set_yscale("log")
    axis.set_xlabel("images per duplicate cluster")
    axis.set_ylabel("clusters (log)")
    axis.set_title(f"Duplicate cluster sizes (n={sizes.size}, largest={maximum})")
    return _save(figure, path)


def plot_image_size_histogram(sizes: Sequence[tuple[int, int]], path: str) -> str:
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    if not sizes:
        _empty(axis, "no images")
        return _save(figure, path)
    labels = [f"{int(w)}x{int(h)}" for w, h in sizes]
    unique, counts = np.unique(np.asarray(labels), return_counts=True)
    order = np.argsort(-counts)
    unique, counts = unique[order], counts[order]
    axis.bar(range(len(unique)), counts, color="#2ca02c")
    axis.set_xticks(range(len(unique)))
    axis.set_xticklabels(unique, rotation=30, ha="right", fontsize=7)
    axis.set_ylabel("images")
    axis.set_title("Image size histogram")
    return _save(figure, path)


# ─────────────────────────────────────────────────────────────────────────────
# Training figures
# ─────────────────────────────────────────────────────────────────────────────


def plot_training_curves(
    history: Mapping[str, Sequence[float]], path: str, best_epoch: Optional[int] = None
) -> str:
    """Four panels: loss, accuracy, macro-F1, learning rate."""
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 7.5))
    panels = [
        (axes[0][0], "loss", [("train_loss", "train"), ("val_loss", "val")], "Loss"),
        (axes[0][1], "acc", [("train_acc", "train"), ("val_acc", "val")], "Accuracy"),
        (axes[1][0], "f1", [("val_macro_f1", "val macro-F1")], "Validation macro-F1"),
        (axes[1][1], "lr", [("lr", "learning rate")], "Learning rate"),
    ]
    for axis, _key, series, title in panels:
        plotted = False
        for key, label in series:
            values = list(history.get(key, []))
            if values:
                axis.plot(epochs[: len(values)], values, label=label)
                plotted = True
        if not plotted:
            _empty(axis, "no data")
            continue
        if best_epoch:
            axis.axvline(best_epoch, ls=":", lw=1.0, color="tab:red")
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.legend(fontsize=8)
    figure.suptitle("Training curves", fontsize=12)
    return _save(figure, path)


def plot_epoch_time(times: Sequence[float], path: str) -> str:
    figure, axis = plt.subplots(figsize=(7.5, 4.0))
    if not times:
        _empty(axis, "no epochs")
        return _save(figure, path)
    epochs = np.arange(1, len(times) + 1)
    axis.plot(epochs, times, marker="o", ms=3, lw=1.2, color="#1f77b4")
    axis.set_xlabel("epoch")
    axis.set_ylabel("seconds")
    axis.set_title(f"Epoch time (mean {np.mean(times):.1f}s, max {np.max(times):.1f}s)")
    return _save(figure, path)


# ─────────────────────────────────────────────────────────────────────────────
# Test-set figures
# ─────────────────────────────────────────────────────────────────────────────


def plot_per_class_f1(
    per_class: Mapping[str, Mapping[str, Any]], class_names: Sequence[str], path: str
) -> str:
    figure, axis = plt.subplots(figsize=(8.0, 4.4))
    values = [
        float(per_class.get(name, {}).get("f1", 0.0) or 0.0) for name in class_names
    ]
    bars = axis.bar(range(len(class_names)), values, color=_COLOURS[: len(class_names)])
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=15, ha="right")
    axis.set_ylabel("F1")
    axis.axhline(float(np.mean(values)), ls=":", color="gray", label=f"macro {np.mean(values):.3f}")
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_title("Per-class F1 (test)")
    axis.legend(fontsize=8)
    return _save(figure, path)


def plot_per_class_pr(
    per_class: Mapping[str, Mapping[str, Any]], class_names: Sequence[str], path: str
) -> str:
    figure, axis = plt.subplots(figsize=(9.0, 4.6))
    positions = np.arange(len(class_names))
    width = 0.38
    precision = [float(per_class.get(name, {}).get("precision", 0.0) or 0.0) for name in class_names]
    recall = [float(per_class.get(name, {}).get("recall", 0.0) or 0.0) for name in class_names]
    axis.bar(positions - width / 2, precision, width=width, label="precision", color="#1f77b4")
    axis.bar(positions + width / 2, recall, width=width, label="recall", color="#ff7f0e")
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(positions)
    axis.set_xticklabels(class_names, rotation=15, ha="right")
    axis.set_ylabel("score")
    axis.set_title("Per-class precision / recall (test)")
    axis.legend(fontsize=8)
    return _save(figure, path)


def _draw_confusion(
    matrix: np.ndarray, class_names: Sequence[str], path: str, title: str, normalized: bool
) -> str:
    figure, axis = plt.subplots(figsize=(6.4, 5.6))
    data = np.asarray(matrix, dtype=np.float64)
    image = axis.imshow(data, cmap="Blues", vmin=0.0, vmax=max(1.0, float(data.max())))
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
    axis.set_yticklabels(class_names, fontsize=8)
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    axis.set_title(title)
    threshold = data.max() * 0.55 if data.size else 0.0
    for row in range(data.shape[0]):
        for column in range(data.shape[1]):
            value = data[row, column]
            text = f"{value:.2f}" if normalized else f"{int(value)}"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    return _save(figure, path)


def plot_confusion_matrix_counts(
    matrix: np.ndarray, class_names: Sequence[str], path: str
) -> str:
    return _draw_confusion(
        matrix, class_names, path, "Confusion matrix (counts, test)", normalized=False
    )


def plot_confusion_matrix_normalized(
    matrix: np.ndarray, class_names: Sequence[str], path: str
) -> str:
    data = np.asarray(matrix, dtype=np.float64)
    row_sums = data.sum(axis=1, keepdims=True)
    normalized = np.divide(data, np.maximum(row_sums, 1.0))
    return _draw_confusion(
        normalized, class_names, path, "Confusion matrix (row-normalised, test)", normalized=True
    )


def plot_roc_per_class(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: Sequence[str], path: str
) -> str:
    from sklearn.metrics import roc_auc_score, roc_curve

    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    columns = min(4, max(1, len(class_names)))
    rows = int(np.ceil(len(class_names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 3.4 * rows), squeeze=False)
    for index, name in enumerate(class_names):
        axis = axes[index // columns][index % columns]
        binary = (y_true == index).astype(np.int64)
        axis.plot([0, 1], [0, 1], "k--", lw=0.7)
        if binary.min() == binary.max():
            _empty(axis, f"{name}\nsingle class in test")
            continue
        fpr, tpr, _ = roc_curve(binary, y_prob[:, index])
        auc = roc_auc_score(binary, y_prob[:, index])
        axis.plot(fpr, tpr, lw=1.4, color=_COLOURS[index % len(_COLOURS)])
        axis.set_title(f"{name}\nAUC {auc:.3f}", fontsize=9)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.01)
        axis.tick_params(labelsize=7)
    for index in range(len(class_names), rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.suptitle("ROC per class (one-vs-rest, test)", fontsize=11)
    return _save(figure, path)


def plot_pr_curves(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: Sequence[str], path: str
) -> str:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    figure, axis = plt.subplots(figsize=(6.8, 5.2))
    for index, name in enumerate(class_names):
        binary = (y_true == index).astype(np.int64)
        if binary.min() == binary.max():
            continue
        precision, recall, _ = precision_recall_curve(binary, y_prob[:, index])
        ap = average_precision_score(binary, y_prob[:, index])
        axis.plot(recall, precision, lw=1.5, label=f"{name} (AP {ap:.3f})")
    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_title("Precision-recall per class (test)")
    axis.legend(fontsize=8)
    return _save(figure, path)


def plot_calibration_reliability(
    y_true: np.ndarray, y_prob: np.ndarray, path: str, n_bins: int = 15
) -> tuple[str, float]:
    """Reliability diagram + expected calibration error. Returns ``(path, ece)``."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    confidence = y_prob.max(axis=1)
    predicted = y_prob.argmax(axis=1)
    correct = (predicted == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres: list[float] = []
    accuracies: list[float] = []
    confidences: list[float] = []
    ece = 0.0
    total = max(len(y_true), 1)
    for bin_index in range(n_bins):
        low, high = edges[bin_index], edges[bin_index + 1]
        mask = (confidence > low) & (confidence <= high)
        count = int(mask.sum())
        if count == 0:
            continue
        accuracy = float(correct[mask].mean())
        mean_confidence = float(confidence[mask].mean())
        centres.append((low + high) / 2.0)
        accuracies.append(accuracy)
        confidences.append(mean_confidence)
        ece += abs(accuracy - mean_confidence) * count / total

    figure, axis = plt.subplots(figsize=(5.8, 5.2))
    axis.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    if centres:
        axis.plot(confidences, accuracies, marker="o", ms=4, lw=1.3, label="model")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("mean predicted confidence")
    axis.set_ylabel("empirical accuracy")
    axis.set_title(f"Reliability diagram (ECE {ece:.4f}, {n_bins} bins)")
    axis.legend(fontsize=8)
    return _save(figure, path), float(ece)


def plot_gradcam_grid_per_class(entries: Sequence[Mapping[str, Any]], path: str) -> str:
    """Two columns per class: original, Grad-CAM overlay.

    ``entries`` items: ``{"class": str, "image": HxWx3, "overlay": HxWx3, "title": str}``.
    """
    if not entries:
        figure, axis = plt.subplots(figsize=(6, 2))
        _empty(axis, "no Grad-CAM entries")
        return _save(figure, path)

    figure, axes = plt.subplots(
        len(entries), 2, figsize=(6.0, 3.0 * len(entries)), squeeze=False
    )
    for row, entry in enumerate(entries):
        image = np.asarray(entry["image"])
        overlay = np.asarray(entry["overlay"])
        left, right = axes[row][0], axes[row][1]
        left.imshow(image, cmap="gray" if image.ndim == 2 else None)
        left.set_title(f"{entry.get('class', '')} — original", fontsize=8)
        right.imshow(overlay)
        right.set_title(str(entry.get("title", "Grad-CAM")), fontsize=8)
        for axis in (left, right):
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle("Grad-CAM per class", fontsize=11)
    return _save(figure, path)


def plot_tta_comparison(results: Mapping[str, Mapping[str, float]], path: str) -> str:
    """Grouped bars of accuracy / macro-F1 / macro-AUC for each evaluation mode."""
    labels = list(results)
    metrics = ("accuracy", "macro_f1", "macro_auc")
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    positions = np.arange(len(labels))
    width = 0.26
    for offset, metric in enumerate(metrics):
        values = [
            float(results[label].get(metric) or 0.0) for label in labels
        ]
        axis.bar(positions + (offset - 1) * width, values, width=width, label=metric)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("score")
    axis.set_title("Single-scale vs TTA (test)")
    axis.legend(fontsize=8)
    return _save(figure, path)


__all__ = [
    "DPI",
    "plot_augmentation_preview",
    "plot_calibration_reliability",
    "plot_class_distribution",
    "plot_confusion_matrix_counts",
    "plot_confusion_matrix_normalized",
    "plot_duplicate_cluster_histogram",
    "plot_epoch_time",
    "plot_gradcam_grid_per_class",
    "plot_image_size_histogram",
    "plot_per_class_f1",
    "plot_per_class_pr",
    "plot_pr_curves",
    "plot_roc_per_class",
    "plot_sample_grid_per_class",
    "plot_split_distribution",
    "plot_training_curves",
    "plot_tta_comparison",
]
