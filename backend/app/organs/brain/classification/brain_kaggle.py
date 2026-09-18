"""
Brain Classification — Kaggle / Standalone Training Entrypoint.

Orchestrates data loading, stratified splitting (by brain location),
class-weighted training with EfficientNetV2-B2, learning rate scheduling,
early stopping, model checkpointing, evaluation, and comprehensive
academic artifact generation (exported as brain_classification_results.zip).

Usage:
    python brain_kaggle.py --data_root /path/to/archive --batch_size 32 --epochs 50
    python brain_kaggle.py --data_root /kaggle/input/brain-tumor-dataset --output_dir /kaggle/working
"""

import argparse
import csv
import logging
import os
import sys
import time
import zipfile

# Ensure local modules can be imported regardless of execution working directory
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
    from dataset import BrainTumorDataset
    from preprocessor import get_train_transform, get_val_transform
    from model import create_model, get_loss_function, get_optimizer, get_scheduler
    import visualization
except ImportError:
    from app.organs.brain.classification.dataset import BrainTumorDataset
    from app.organs.brain.classification.preprocessor import get_train_transform, get_val_transform
    from app.organs.brain.classification.model import create_model, get_loss_function, get_optimizer, get_scheduler
    from app.organs.brain.classification import visualization

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("brain_kaggle")


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    for inputs, labels, _ in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total):.3f}"})

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch, return_preds=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Val]  ", leave=False)
    with torch.no_grad():
        for inputs, labels, _ in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            probs = F.softmax(outputs, dim=1)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total):.3f}"})

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)

    if return_preds:
        return epoch_loss, epoch_acc, all_labels, all_preds, all_probs
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train OpenMed Brain Tumor Classification Model")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/honeypot/Projects/FAST_API/OpenMed/backend/datasets/brain/archive",
        help="Path to brain dataset archive directory or DATA.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for checkpoints, logs, and results zip",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (default: 4)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    artifacts_dir = os.path.join(args.output_dir, "artifacts")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("━" * 60)
    logger.info("  OpenMed Brain Classification — Kaggle Training")
    logger.info("━" * 60)
    logger.info(f"  Device           : {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU Name         : {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU Memory       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info(f"  Dataset Path     : {args.data_root}")
    logger.info(f"  Output Directory : {args.output_dir}")
    logger.info(f"  Batch Size       : {args.batch_size}")
    logger.info(f"  Max Epochs       : {args.epochs}")
    logger.info("━" * 60)

    # 1. Dataset & Stratified Splits
    logger.info("Loading DATA.json and computing location-stratified splits...")
    train_idx, val_idx, test_idx = BrainTumorDataset.get_stratified_splits(args.data_root)

    # Resolve native transforms via timm config
    base_model = create_model(num_classes=39, pretrained=True)
    train_transform = get_train_transform(base_model)
    val_transform = get_val_transform(base_model)

    train_dataset = BrainTumorDataset(args.data_root, transform=train_transform, split_indices=train_idx)
    val_dataset = BrainTumorDataset(args.data_root, transform=val_transform, split_indices=val_idx)
    test_dataset = BrainTumorDataset(args.data_root, transform=val_transform, split_indices=test_idx)

    class_names = train_dataset.CLASS_NAMES
    num_classes = len(class_names)
    logger.info(f"Classes catalogued: {num_classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # 2. Model, Loss, Optimizer, Scheduler Setup
    model = base_model.to(device)
    class_weights = BrainTumorDataset.compute_class_weights(train_dataset.samples, num_classes)
    criterion = get_loss_function(class_weights, device)

    optimizer = get_optimizer(model, lr=args.lr)
    steps_per_epoch = len(train_loader)
    scheduler = get_scheduler(optimizer, num_epochs=args.epochs, steps_per_epoch=steps_per_epoch)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    log_csv_path = os.path.join(args.output_dir, "training_log.csv")
    csv_file = open(log_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "elapsed_time"])

    best_model_path = os.path.join(checkpoints_dir, "brain_best_model.pth")

    # 3. Training Loop
    logger.info("Starting training loop with early stopping...")
    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)

        ep_time = time.time() - ep_start
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        csv_writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", f"{current_lr:.6f}", f"{ep_time:.2f}"])
        csv_file.flush()

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"Time: {ep_time:.1f}s"
        )

        # Checkpointing (best model)
        if val_loss < best_val_loss or val_acc > best_val_acc:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            torch.save(model.state_dict(), best_model_path)
            logger.info(f" ⭐ Best model checkpoint saved (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered at epoch {epoch} (patience={args.patience})")
            break

    csv_file.close()
    total_elapsed = time.time() - start_total_time
    logger.info(f"Training finished in {total_elapsed/60:.2f} minutes.")

    # 4. Evaluation on Test Set & Artifact Generation
    logger.info("Evaluating best model on held-out test split...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc, y_true, y_pred, y_probs = validate(
        model, test_loader, criterion, device, 0, return_preds=True
    )
    logger.info(f"Test Set Evaluation — Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    logger.info("Generating comprehensive academic artifacts...")

    # 4.1 Training loss & accuracy curves
    visualization.plot_training_curves(history, artifacts_dir)

    # 4.2 Normalized Confusion Matrix
    visualization.plot_confusion_matrix(y_true, y_pred, class_names, artifacts_dir)

    # 4.3 Class distribution plot across splits
    train_labels = [s["metadata"]["class"] for s in train_dataset.samples]
    val_labels = [s["metadata"]["class"] for s in val_dataset.samples]
    test_labels = [s["metadata"]["class"] for s in test_dataset.samples]

    train_label_indices = [train_dataset.class_to_idx[l] for l in train_labels]
    val_label_indices = [val_dataset.class_to_idx[l] for l in val_labels]
    test_label_indices = [test_dataset.class_to_idx[l] for l in test_labels]
    visualization.plot_class_distribution(train_label_indices, val_label_indices, test_label_indices, class_names, artifacts_dir)

    # 4.4 Sample prediction grid (4x4)
    sample_images, sample_labels, _ = next(iter(test_loader))
    sample_images = sample_images[:16]
    sample_labels = sample_labels[:16]
    with torch.no_grad():
        out = model(sample_images.to(device))
        _, preds = out.max(1)
        sample_preds = preds.cpu().numpy()
    visualization.plot_sample_predictions(sample_images, sample_labels.numpy(), sample_preds, class_names, artifacts_dir)

    # 4.5 Grad-CAM Heatmap Visualization
    visualization.plot_gradcam_samples(model, test_dataset, device, class_names, artifacts_dir)

    # 4.6 & 4.7 ROC & Precision-Recall Curves
    visualization.plot_roc_curves(y_true, np.array(y_probs), class_names, artifacts_dir)
    visualization.plot_pr_curves(y_true, np.array(y_probs), class_names, artifacts_dir)

    # 4.8 Detailed Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    report_path = os.path.join(artifacts_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("OpenMed Brain Classification — Academic Evaluation Report\n")
        f.write(f"Model: EfficientNetV2-B2 (tf_efficientnetv2_b2.in1k)\n")
        f.write(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}\n\n")
        f.write(report)

    # 5. Export everything as .zip
    zip_path = os.path.join(args.output_dir, "brain_classification_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        if os.path.exists(log_csv_path):
            zipf.write(log_csv_path, "training_log.csv")
        if os.path.exists(best_model_path):
            zipf.write(best_model_path, "brain_best_model.pth")
        for root, _, files in os.walk(artifacts_dir):
            for file in files:
                full_p = os.path.join(root, file)
                rel_p = os.path.relpath(full_p, args.output_dir)
                zipf.write(full_p, rel_p)

    logger.info("━" * 60)
    logger.info(f"✅ Training and evaluation complete!")
    logger.info(f"   Model checkpoint saved to : {best_model_path}")
    logger.info(f"   Artifacts generated in    : {artifacts_dir}")
    logger.info(f"   Bundle exported to        : {zip_path}")
    logger.info("━" * 60)


if __name__ == "__main__":
    main()
