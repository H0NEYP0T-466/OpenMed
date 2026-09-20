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
from timm.data import Mixup
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import visualization
    from dataset import BrainTumorDataset, discover_dataset
    from label_space import label_space_path_for, save_label_space
    from model import (
        MODEL_TAG,
        create_model,
        get_loss_function,
        get_optimizer,
        get_scheduler,
        weighted_soft_target_cross_entropy,
    )
    from preprocessor import (
        build_class_aware_transforms,
        get_train_transform,
        get_val_transform,
    )
except ImportError:
    from . import visualization
    from .dataset import BrainTumorDataset, discover_dataset
    from .label_space import label_space_path_for, save_label_space
    from .model import (
        MODEL_TAG,
        create_model,
        get_loss_function,
        get_optimizer,
        get_scheduler,
        weighted_soft_target_cross_entropy,
    )
    from .preprocessor import (
        build_class_aware_transforms,
        get_train_transform,
        get_val_transform,
    )

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


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epoch,
    mixup_fn=None,
    class_weights=None,
    ema=None,
):
    """One training pass; MixUp/CutMix batches optimise the weighted soft-target loss.

    Accuracy is logged against the original hard labels so the curve stays
    comparable across runs even though the optimisation target is a mixture.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=f"Epoch {epoch:02d} [train]", leave=False)
    for inputs, labels, _ in progress:
        inputs, hard_labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        if mixup_fn is not None and inputs.size(0) > 1:
            inputs, soft_targets = mixup_fn(inputs, hard_labels)
            outputs = model(inputs)
            loss = weighted_soft_target_cross_entropy(
                outputs, soft_targets, class_weights
            )
        else:
            outputs = model(inputs)
            loss = criterion(outputs, hard_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        batch = inputs.size(0)
        running_loss += float(loss.item()) * batch
        correct += int(outputs.argmax(1).eq(hard_labels).sum().item())
        total += batch
        progress.set_postfix(
            loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.3f}"
        )

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, return_preds=False, tag="val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    progress = tqdm(loader, desc=f"Epoch {epoch:02d} [{tag}    ]", leave=False)
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
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--drop_path", type=float, default=0.15,
        help="Stochastic depth rate (primary EfficientNet regulariser)",
    )
    parser.add_argument("--mixup_alpha", type=float, default=0.1)
    parser.add_argument("--cutmix_alpha", type=float, default=0.2)
    parser.add_argument(
        "--mixup_prob", type=float, default=0.3,
        help="Probability that a batch is mixed; 0.3 for gentle regularization",
    )
    parser.add_argument(
        "--auto_augment", default="rand-m5-mstd0.5-inc1",
        help="timm RandAugment spec for standard classes; pass '' to disable",
    )
    parser.add_argument(
        "--minority_augment", default="rand-m7-mstd0.5-inc1",
        help="Stronger RandAugment spec for minority/lower-sample classes",
    )
    parser.add_argument(
        "--ema_decay", type=float, default=0.999,
        help="Exponential-moving-average decay for eval weights; 0 disables",
    )
    parser.add_argument(
        "--pct_start", type=float, default=0.2,
        help="One-cycle warmup share of total steps",
    )
    return parser.parse_args(argv)


def write_split_manifest(path: str, dataset_root: str, splits: dict[str, list[int]]) -> int:
    """Record every sample with its split and source-group assignment.

    The `group_id` column makes the manifest self-auditable: a reviewer can
    verify split isolation from this file alone, without re-deriving groups
    under whatever grouping code exists later.
    """
    view = discover_dataset(dataset_root)
    keys = view.keys
    labels = [view.labels[key] for key in keys]
    group_ids = BrainTumorDataset.build_group_ids(
        view.manifest_path,
        view.images_base,
        keys,
        labels=labels,
        cache_dir=view.cache_dir,
    )
    group_of = dict(zip(keys, group_ids, strict=True))

    written = 0
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["relative_path", "class", "tumor_type", "sequence", "split", "group_id"]
        )
        for name, indices in splits.items():
            for index in indices:
                key = keys[index]
                meta = view.metadata[key]
                writer.writerow(
                    [
                        key,
                        meta.get("class", view.labels[key]),
                        meta.get("tumor_type", view.labels[key]),
                        meta.get("sequence", ""),
                        name,
                        group_of[key],
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

    base_model = create_model(
        num_classes=num_classes,
        pretrained=True,
        model_tag=MODEL_TAG,
        drop_path_rate=args.drop_path,
    )
    MINORITY_CLASSES = (
        "Germ Cell Tumors",
        "Mesenchymal (Non-Meningothelial Tumors)",
        "Mixed Neuronal and Neuronal-Glial Tumors",
        "Medulloblastoma",
    )
    default_transform, minority_transforms = build_class_aware_transforms(
        base_model,
        class_names=class_names,
        minority_classes=MINORITY_CLASSES,
        base_augment=args.auto_augment or None,
        minority_augment=args.minority_augment,
    )
    train_dataset = BrainTumorDataset(
        args.data_root,
        transform=default_transform,
        transforms_by_class=minority_transforms,
        split_indices=train_idx,
    )
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
    ema = ModelEmaV2(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    weights = BrainTumorDataset.compute_class_weights(train_dataset.samples, class_names)
    criterion = get_loss_function(weights, device)
    class_weight_tensor = weights.to(device)
    mixup_fn = None
    if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=args.mixup_prob,
            switch_prob=0.5,
            label_smoothing=0.1,
            num_classes=num_classes,
        )
    optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(
        optimizer,
        num_epochs=args.epochs,
        steps_per_epoch=max(len(train_loader), 1),
        pct_start=args.pct_start,
    )
    logger.info(
        "Regularisation: drop_path=%.2f mixup=%.2f cutmix=%.2f auto_augment=%s "
        "ema_decay=%s label_smoothing=0.1",
        args.drop_path, args.mixup_alpha, args.cutmix_alpha,
        args.auto_augment or "off", args.ema_decay or "off",
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
        writer.writerow([
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
            "val_loss_ema", "val_acc_ema", "lr", "seconds",
        ])

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, epoch,
                mixup_fn=mixup_fn, class_weights=class_weight_tensor, ema=ema,
            )
            raw_loss, raw_acc = validate(model, val_loader, criterion, device, epoch)
            ema_loss = ema_acc = None
            monitor_state = acc_state = model.state_dict()
            val_loss, val_acc = raw_loss, raw_acc
            if ema is not None:
                ema_loss, ema_acc = validate(
                    ema.module, val_loader, criterion, device, epoch, tag="val-ema"
                )
                # The EMA weights usually generalise better; whichever variant
                # wins per metric is the one that gets checkpointed.
                if ema_loss < raw_loss:
                    val_loss = ema_loss
                    monitor_state = ema.module.state_dict()
                if ema_acc > raw_acc:
                    val_acc = ema_acc
                    acc_state = ema.module.state_dict()
            elapsed = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            writer.writerow([
                epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                f"{val_loss:.4f}", f"{val_acc:.4f}",
                f"{ema_loss:.4f}" if ema_loss is not None else "",
                f"{ema_acc:.4f}" if ema_acc is not None else "",
                f"{current_lr:.6f}", f"{elapsed:.2f}",
            ])
            csv_handle.flush()

            markers = []
            improved_monitor = False
            if val_loss < best_monitor:
                best_monitor = val_loss
                improved_monitor = True
                torch.save(monitor_state, best_path)
                save_label_space(best_path, class_names, model_tag=MODEL_TAG)
                source = "raw" if (ema is None or val_loss == raw_loss) else "ema"
                markers.append(f"best {MONITOR}={val_loss:.4f} ({source})")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(acc_state, best_acc_path)
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
        "regularization": {
            "drop_path": args.drop_path,
            "mixup_alpha": args.mixup_alpha,
            "cutmix_alpha": args.cutmix_alpha,
            "auto_augment": args.auto_augment or None,
            "ema_decay": args.ema_decay or None,
            "label_smoothing": 0.1 if mixup_fn is not None else 0.0,
            "weight_decay_exempt": "norm+bias",
        },
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
            "Regularisation": (
                f"drop_path={args.drop_path}, mixup={args.mixup_alpha}, "
                f"cutmix={args.cutmix_alpha}, randaugment={args.auto_augment or 'off'}, "
                f"ema={args.ema_decay or 'off'}, label_smoothing={0.1 if mixup_fn else 0.0}"
            ),
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
