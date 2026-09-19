"""Training artefact plotting: curves, matrices, distributions and Grad-CAM panels."""

from __future__ import annotations

import logging
import os
import random
from collections import Counter
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

try:
    from model import GradCAM, overlay_cam_on_image, resolve_gradcam_layer
except ImportError:
    from .model import GradCAM, overlay_cam_on_image, resolve_gradcam_layer

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ARTIFACT_SEED = 42


def _inverse_normalize(tensor: Any, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0) if hasattr(tensor, "numpy") else np.asarray(tensor)
    array = array * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
    return np.clip(array, 0.0, 1.0)


def _save(figure: Any, save_dir: str, filename: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    target = os.path.join(save_dir, filename)
    figure.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_training_curves(history: dict[str, Sequence[float]], save_dir: str) -> list[str]:
    """Plot loss and accuracy curves for train and validation."""
    if not history.get("train_loss"):
        logger.warning("No training history to plot.")
        return []

    epochs = range(1, len(history["train_loss"]) + 1)
    written = []

    figure, axes = plt.subplots(figsize=(10, 5))
    axes.plot(epochs, history["train_loss"], label="Train loss")
    axes.plot(epochs, history["val_loss"], label="Validation loss")
    axes.set_title("Loss per epoch")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")
    axes.grid(True, alpha=0.3)
    axes.legend()
    written.append(_save(figure, save_dir, "loss_curves.png"))

    figure, axes = plt.subplots(figsize=(10, 5))
    axes.plot(epochs, history["train_acc"], label="Train accuracy")
    axes.plot(epochs, history["val_acc"], label="Validation accuracy")
    axes.set_title("Accuracy per epoch")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Accuracy")
    axes.set_ylim(0.0, 1.05)
    axes.grid(True, alpha=0.3)
    axes.legend()
    written.append(_save(figure, save_dir, "accuracy_curves.png"))

    return written


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    save_dir: str,
) -> list[str]:
    """Row-normalised confusion matrix over the classes actually present."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    present = np.intersect1d(np.unique(np.concatenate([y_true, y_pred])), np.arange(len(class_names)))
    if present.size == 0:
        logger.warning("Confusion matrix skipped: no predicted labels.")
        return []

    names = [class_names[i] for i in present]
    matrix = confusion_matrix(y_true, y_pred, labels=present)
    totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0)

    size = max(10.0, 0.42 * present.size)
    figure, axes = plt.subplots(figsize=(size, size))
    sns.heatmap(
        normalised,
        annot=present.size <= 12,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        xticklabels=names,
        yticklabels=names,
        ax=axes,
        square=True,
    )
    axes.set_title("Row-normalised confusion matrix (recalled class shares)")
    axes.set_ylabel("True label")
    axes.set_xlabel("Predicted label")
    plt.setp(axes.get_xticklabels(), rotation=90)
    plt.setp(axes.get_yticklabels(), rotation=0)

    return [_save(figure, save_dir, "confusion_matrix.png")]


def plot_class_distribution(
    train_labels: Sequence[int],
    val_labels: Sequence[int],
    test_labels: Sequence[int],
    class_names: Sequence[str],
    save_dir: str,
) -> list[str]:
    """Sample counts per class, split by assignment."""
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    test_counts = Counter(test_labels)

    indices = np.arange(len(class_names))
    width = 0.26
    train = np.array([train_counts.get(i, 0) for i in indices], dtype=float)
    val = np.array([val_counts.get(i, 0) for i in indices], dtype=float)
    test = np.array([test_counts.get(i, 0) for i in indices], dtype=float)

    figure, axes = plt.subplots(figsize=(max(14.0, 0.5 * len(class_names)), 8))
    axes.bar(indices - width, train, width, label=f"Train (n={int(train.sum())})")
    axes.bar(indices, val, width, label=f"Val (n={int(val.sum())})")
    axes.bar(indices + width, test, width, label=f"Test (n={int(test.sum())})")

    axes.set_yscale("log")
    axes.set_ylabel("Samples (log scale)")
    axes.set_title("Class distribution by split")
    axes.set_xticks(indices)
    axes.set_xticklabels(class_names, rotation=90)
    axes.legend()
    axes.grid(True, axis="y", alpha=0.3)

    return [_save(figure, save_dir, "class_distribution.png")]


def plot_sample_predictions(
    images: Any,
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
    class_names: Sequence[str],
    save_dir: str,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    grid: tuple[int, int] = (4, 4),
) -> list[str]:
    """Grid of predictions with true/predicted captions, red where wrong."""
    rows, columns = grid
    count = min(len(images), rows * columns)
    if count == 0:
        logger.warning("No samples available for the prediction grid.")
        return []

    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 4.0 * rows))
    for slot, axes_cell in enumerate(np.atleast_1d(axes).ravel()):
        if slot >= count:
            axes_cell.axis("off")
            continue

        picture = _inverse_normalize(images[slot], mean, std)
        true_index, pred_index = int(true_labels[slot]), int(pred_labels[slot])
        correct = true_index == pred_index

        axes_cell.imshow(picture)
        axes_cell.set_title(
            f"True: {class_names[true_index]}\nPred: {class_names[pred_index]}",
            color="green" if correct else "red",
            fontsize=7,
        )
        axes_cell.axis("off")

    figure.suptitle("Sample predictions", y=1.0)
    figure.tight_layout()
    return [_save(figure, save_dir, "sample_predictions.png")]


def plot_gradcam_samples(
    model: Any,
    dataset: Any,
    device: Any,
    class_names: Sequence[str],
    save_dir: str,
    num_samples: int = 8,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    seed: int = ARTIFACT_SEED,
) -> list[str]:
    """Grad-CAM overlays for a reproducible sample of a dataset split."""
    available = len(dataset)
    if available == 0:
        logger.warning("Grad-CAM skipped: empty dataset.")
        return []

    rng = random.Random(seed)
    taken = rng.sample(range(available), min(num_samples, available))
    rows, columns = 2, 4

    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 4.0 * rows))
    flat = np.atleast_1d(axes).ravel()

    try:
        with GradCAM(model, resolve_gradcam_layer(model)) as grad_cam:
            for slot, index in enumerate(taken):
                image_tensor, label, _ = dataset[index]
                tensor = image_tensor.unsqueeze(0).to(device)
                cam = grad_cam.generate(tensor, int(label))
                picture = _inverse_normalize(image_tensor, mean, std)
                overlay = overlay_cam_on_image((picture * 255).astype(np.uint8), cam)

                flat[slot].imshow(overlay)
                flat[slot].set_title(class_names[int(label)], fontsize=8)
                flat[slot].axis("off")
    except RuntimeError as exc:
        logger.error("Grad-CAM generation failed: %s", exc)
        plt.close(figure)
        return []

    for slot in range(len(taken), flat.size):
        flat[slot].axis("off")

    figure.suptitle("Grad-CAM attribution for the true class", y=1.0)
    figure.tight_layout()
    return [_save(figure, save_dir, "gradcam_samples.png")]


def _one_hot(y_true: Sequence[int], n_classes: int) -> np.ndarray:
    encoded = np.zeros((len(y_true), n_classes), dtype=float)
    for row, column in enumerate(np.asarray(y_true, dtype=int)):
        if 0 <= column < n_classes:
            encoded[row, column] = 1.0
    return encoded


def plot_roc_curves(
    y_true: Sequence[int],
    y_scores: np.ndarray,
    class_names: Sequence[str],
    save_dir: str,
    max_per_class: int = 8,
) -> list[str]:
    """Macro-averaged ROC with the highest-AUC classes highlighted."""
    n_classes = len(class_names)
    encoded = _one_hot(y_true, n_classes)

    per_class: dict[int, float] = {}
    figure, axes = plt.subplots(figsize=(9, 8))

    for index in range(n_classes):
        if encoded[:, index].sum() == 0:
            continue
        try:
            false_positive, true_positive, _ = roc_curve(encoded[:, index], y_scores[:, index])
        except ValueError:
            continue
        score = auc(false_positive, true_positive)
        per_class[index] = score
        if len(per_class) <= max_per_class:
            axes.plot(
                false_positive,
                true_positive,
                lw=1,
                alpha=0.55,
                label=f"{class_names[index]} (AUC {score:.2f})",
            )

    try:
        micro = roc_auc_score(encoded, y_scores, average="micro", multi_class="ovr")
    except ValueError:
        micro = None

    if per_class:
        axes.plot([0, 1], [0, 1], "k--", lw=1.2, label="Chance")
        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(0.0, 1.02)
        axes.set_xlabel("False positive rate")
        axes.set_ylabel("True positive rate")
        title = f"ROC - mean per-class AUC {np.mean(list(per_class.values())):.3f}"
        if micro is not None:
            title += f", micro AUC {micro:.3f}"
        axes.set_title(title)
        axes.legend(fontsize=7, loc="lower right")
        return [_save(figure, save_dir, "roc_curves.png")]

    plt.close(figure)
    logger.warning("ROC curves skipped: no computable classes.")
    return []


def plot_pr_curves(
    y_true: Sequence[int],
    y_scores: np.ndarray,
    class_names: Sequence[str],
    save_dir: str,
) -> list[str]:
    """Per-class precision-recall curves for classes with support."""
    n_classes = len(class_names)
    encoded = _one_hot(y_true, n_classes)

    figure, axes = plt.subplots(figsize=(9, 7))
    plotted = 0
    for index in range(n_classes):
        support = encoded[:, index].sum()
        if support == 0:
            continue
        precision, recall, _ = precision_recall_curve(encoded[:, index], y_scores[:, index])
        area = auc(recall, precision)
        axes.plot(recall, precision, lw=1, alpha=0.55, label=f"{class_names[index]} (AP {area:.2f})")
        plotted += 1

    if plotted == 0:
        plt.close(figure)
        logger.warning("PR curves skipped: no classes with support.")
        return []

    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.set_ylim(0.0, 1.02)
    axes.set_title(f"Precision-recall ({plotted} classes with support)")
    axes.legend(fontsize=6, loc="lower left", ncol=2)
    axes.grid(True, alpha=0.25)

    return [_save(figure, save_dir, "pr_curves.png")]


def write_evaluation_report(
    path: str,
    test_accuracy: float,
    test_loss: float,
    report_text: str,
    provenance: Optional[dict[str, Any]] = None,
) -> str:
    """Write the classification report with the provenance needed to interpret it."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        handle.write("OpenMed Brain Classification - Evaluation Report\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"Test accuracy: {test_accuracy:.4f}\n")
        handle.write(f"Test loss    : {test_loss:.4f}\n")
        for key, value in (provenance or {}).items():
            handle.write(f"{key:<13}: {value}\n")
        handle.write(
            "\nProtocol: splits are grouped by source scan and byte-identical\n"
            "duplicates are collapsed, so no image group appears in two splits.\n"
        )
        handle.write("\n" + report_text + "\n")
    return path
