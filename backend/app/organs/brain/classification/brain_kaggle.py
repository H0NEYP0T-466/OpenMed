"""
Brain classification training entrypoint.

Runs the leak-free grouped split, trains EfficientNetV2-B2 with class-weighted
cross entropy under a one-cycle schedule, then writes checkpoints, an auditable
split manifest and evaluation artefacts.

Usage:
    python brain_kaggle.py --data_root /path/to/archive --output_dir /kaggle/working
    OPENMED_BRAIN_DATASET=/path/to/archive python brain_kaggle.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
import zipfile

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import visualization
    from dataset import BrainTumorDataset
    from label_space import label_space_path_for, save_label_space
    from model import (
        MODEL_TAG,
        create_model,
        get_loss_function,
        get_optimizer,
        get_scheduler,
    )
    from preprocessor import get_train_transform, get_val_transform
except ImportError:
    from . import visualization
    from .dataset import BrainTumorDataset
    from .label_space import label_space_path_for, save_label_space
    from .model import (
        MODEL_TAG,
        create_model,
        get_loss_function,
        get_optimizer,
        get_scheduler,
    )
    from .preprocessor import get_train_transform, get_val_transform

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("brain_kaggle")

MONITOR = "val_loss"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=f"Epoch {epoch:02d} [train]", leave=False)
    for inputs, labels, _ in progress:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch = inputs.size(0)
        running_loss += float(loss.item()) * batch
        correct += int(outputs.argmax(1).eq(labels).sum().item())
        total += batch
        progress.set_postfix(
            loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.3f}"
        )

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, return_preds=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    progress = tqdm(loader, desc=f"Epoch {epoch:02d} [val]  ", leave=False)
    for inputs, labels, _ in progress:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        batch = inputs.size(0)
        running_loss += float(loss.item()) * batch
        predicted = outputs.argmax(1)
        correct += int(predicted.eq(labels).sum().item())
        total += batch

        if return_preds:
            predictions.append(predicted.cpu().numpy())
            targets.append(labels.cpu().numpy())
            probabilities.append(F.softmax(outputs, dim=1).cpu().numpy())
        progress.set_postfix(
            loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.3f}"
        )

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    if not return_preds:
        return epoch_loss, epoch_acc

    return (
        epoch_loss,
        epoch_acc,
        np.concatenate(targets),
        np.concatenate(predictions),
        np.vstack(probabilities),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the OpenMed brain tumour classifier")
    parser.add_argument(
        "--data_root",
        default=os.getenv("OPENMED_BRAIN_DATASET", "datasets/brain/archive"),
        help="Dataset directory, DATA.json or archive zip (default: $OPENMED_BRAIN_DATASET)",
    )
    parser.add_argument("--output_dir", default=os.getenv("OPENMED_BRAIN_OUTPUT", "."))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def write_split_manifest(path: str, dataset_root: str, splits: dict[str, list[int]]) -> int:
    """Record every sample with its split assignment so results stay auditable."""
    json_path, _ = BrainTumorDataset.locate_data_and_images(dataset_root)
    with open(json_path) as handle:
        raw = json.load(handle)
    keys = sorted(key for key in raw if not key.endswith(BrainTumorDataset.MASK_SUFFIX))

    written = 0
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "class", "tumor_type", "sequence", "split"])
        for name, indices in splits.items():
            for index in indices:
                key = keys[index]
                meta = raw[key]
                writer.writerow(
                    [
                        key,
                        meta["class"],
                        meta.get("tumor_type", ""),
                        meta.get("sequence", ""),
                        name,
                    ]
                )
                written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    artifacts_dir = os.path.join(args.output_dir, "artifacts")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 68)
    logger.info("  OpenMed Brain Classification - training run")
    logger.info("=" * 68)
    logger.info("  device=%s  seed=%d  batch=%d  epochs=%d  lr=%g",
                device, args.seed, args.batch_size, args.epochs, args.lr)
    if torch.cuda.is_available():
        logger.info("  gpu=%s (%.1f GB)", torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)
    logger.info("  data_root=%s", args.data_root)
    logger.info("  output_dir=%s", args.output_dir)
    logger.info("=" * 68)

    train_idx, val_idx, test_idx = BrainTumorDataset.get_stratified_splits(
        args.data_root, seed=args.seed
    )
    manifest_path = os.path.join(artifacts_dir, "split_manifest.csv")
    rows = write_split_manifest(
        manifest_path, args.data_root, {"train": train_idx, "val": val_idx, "test": test_idx}
    )
    logger.info("Split manifest with %d rows written to %s", rows, manifest_path)

    probe = BrainTumorDataset(args.data_root, transform=None, split_indices=train_idx)
    class_names = probe.class_names
    num_classes = len(class_names)

    base_model = create_model(num_classes=num_classes, pretrained=True, model_tag=MODEL_TAG)
    train_dataset = BrainTumorDataset(args.data_root, transform=get_train_transform(base_model), split_indices=train_idx)
    val_dataset = BrainTumorDataset(args.data_root, transform=get_val_transform(base_model), split_indices=val_idx)
    test_dataset = BrainTumorDataset(args.data_root, transform=get_val_transform(base_model), split_indices=test_idx)

    def loader_for(dataset: BrainTumorDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle,
            collate_fn=BrainTumorDataset.collate_fn,
            persistent_workers=args.num_workers > 0,
        )

    train_loader = loader_for(train_dataset, shuffle=True)
    val_loader = loader_for(val_dataset, shuffle=False)
    test_loader = loader_for(test_dataset, shuffle=False)

    model = base_model.to(device)
    weights = BrainTumorDataset.compute_class_weights(train_dataset.samples, class_names)
    criterion = get_loss_function(weights, device)
    optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(
        optimizer, num_epochs=args.epochs, steps_per_epoch=max(len(train_loader), 1)
    )

    best_path = os.path.join(checkpoints_dir, "brain_best_model.pth")
    best_acc_path = os.path.join(checkpoints_dir, "brain_best_acc.pth")
    log_path = os.path.join(args.output_dir, "training_log.csv")

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
    }
    best_monitor = float("inf")
    best_acc = 0.0
    patience_counter = 0
    started = time.time()

    logger.info("Training with %s as the checkpoint monitor.", MONITOR)
    with open(log_path, "w", newline="") as csv_handle:
        writer = csv.writer(csv_handle)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"])

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, epoch
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
            elapsed = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            writer.writerow([
                epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                f"{val_loss:.4f}", f"{val_acc:.4f}", f"{current_lr:.6f}", f"{elapsed:.2f}",
            ])
            csv_handle.flush()

            markers = []
            improved_monitor = False
            if val_loss < best_monitor:
                best_monitor = val_loss
                improved_monitor = True
                torch.save(model.state_dict(), best_path)
                save_label_space(best_path, class_names, model_tag=MODEL_TAG)
                markers.append(f"best {MONITOR}={val_loss:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_acc_path)
                save_label_space(best_acc_path, class_names, model_tag=MODEL_TAG)
                markers.append(f"best val_acc={val_acc:.4f}")

            logger.info(
                "Epoch %02d/%02d | train %.4f / %.4f | val %.4f / %.4f | %.1fs %s",
                epoch, args.epochs, train_loss, train_acc, val_loss, val_acc,
                elapsed, ("| " + ", ".join(markers)) if markers else "",
            )

            if improved_monitor:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d on %s)",
                        epoch, args.patience, MONITOR,
                    )
                    break

    logger.info("Training finished in %.2f minutes.", (time.time() - started) / 60)

    if not os.path.isfile(best_path):
        logger.error("No checkpoint was ever saved; aborting evaluation.")
        return 1

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()

    test_loss, test_acc, y_true, y_pred, y_probs = validate(
        model, test_loader, criterion, device, 0, return_preds=True
    )
    logger.info("Test (monitor checkpoint) loss=%.4f acc=%.4f", test_loss, test_acc)

    report = classification_report(
        y_true, y_pred, labels=range(num_classes),
        target_names=list(class_names), digits=4, zero_division=0,
    )

    mean_class_auc = None
    try:
        from sklearn.metrics import roc_auc_score

        encoded = np.zeros((len(y_true), num_classes))
        encoded[np.arange(len(y_true)), y_true] = 1.0
        supported = encoded.sum(axis=0) > 0
        if supported.sum() > 1:
            mean_class_auc = float(
                roc_auc_score(encoded[:, supported], y_probs[:, supported], average="macro")
            )
    except (ValueError, RuntimeError) as exc:
        logger.warning("Macro AUC unavailable: %s", exc)

    written = []
    written += visualization.plot_training_curves(history, artifacts_dir)
    written += visualization.plot_confusion_matrix(y_true, y_pred, class_names, artifacts_dir)
    written += visualization.plot_class_distribution(
        [train_dataset.class_to_idx[s["metadata"]["class"]] for s in train_dataset.samples],
        [val_dataset.class_to_idx[s["metadata"]["class"]] for s in val_dataset.samples],
        [test_dataset.class_to_idx[s["metadata"]["class"]] for s in test_dataset.samples],
        class_names,
        artifacts_dir,
    )

    sample_images, sample_labels, _ = next(iter(test_loader))
    take = min(16, len(sample_images))
    with torch.no_grad():
        sample_preds = model(sample_images[:take].to(device)).argmax(1).cpu().numpy()
    written += visualization.plot_sample_predictions(
        sample_images[:take], sample_labels.numpy()[:take], sample_preds, class_names, artifacts_dir
    )
    written += visualization.plot_gradcam_samples(
        model, test_dataset, device, class_names, artifacts_dir, seed=args.seed
    )
    written += visualization.plot_roc_curves(y_true, y_probs, class_names, artifacts_dir)
    written += visualization.plot_pr_curves(y_true, y_probs, class_names, artifacts_dir)

    metrics = {
        "model_tag": MODEL_TAG,
        "num_classes": num_classes,
        "seed": args.seed,
        "monitor": MONITOR,
        "best_val_loss": round(best_monitor, 6),
        "best_val_acc": round(best_acc, 6),
        "test_loss": round(float(test_loss), 6),
        "test_acc": round(float(test_acc), 6),
        "macro_auc_ovr": round(mean_class_auc, 6) if mean_class_auc is not None else None,
        "splits": {
            "train": len(train_idx), "val": len(val_idx), "test": len(test_idx)
        },
        "grouping": "source-scan groups with byte-identical duplicates collapsed",
    }
    with open(os.path.join(artifacts_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)

    save_label_space(best_path, class_names, model_tag=MODEL_TAG, metrics=metrics)
    label_file = label_space_path_for(best_path)

    report_path = visualization.write_evaluation_report(
        os.path.join(artifacts_dir, "classification_report.txt"),
        test_acc, test_loss, report,
        provenance={
            "Model": MODEL_TAG,
            "Monitor": MONITOR,
            "Seed": args.seed,
            "Train/Val/Test": f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}",
            "Macro AUC (OvR)": f"{mean_class_auc:.4f}" if mean_class_auc is not None else "n/a",
        },
    )

    zip_path = os.path.join(args.output_dir, "brain_classification_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, arcname in (
            (log_path, "training_log.csv"),
            (best_path, "brain_best_model.pth"),
            (best_acc_path, "brain_best_acc.pth"),
            (label_file, "brain_best_model.label_space.json"),
        ):
            if os.path.isfile(path):
                archive.write(path, arcname)
        for root, _, files in os.walk(artifacts_dir):
            for name in files:
                full = os.path.join(root, name)
                archive.write(full, os.path.relpath(full, args.output_dir))

    logger.info("=" * 68)
    logger.info("  Run complete")
    logger.info("  checkpoint : %s", best_path)
    logger.info("  label file : %s", label_file)
    logger.info("  report     : %s", report_path)
    logger.info("  artefacts  : %d figures in %s", len(written), artifacts_dir)
    logger.info("  bundle     : %s", zip_path)
    logger.info("  test acc   : %.4f (grouped, leak-free split)", test_acc)
    logger.info("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
