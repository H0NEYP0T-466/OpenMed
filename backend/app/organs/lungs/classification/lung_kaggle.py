"""
Z-Net training on NIH ChestX-ray14 — self-contained Kaggle script.

Replicates Ahmad et al., Diagnostics 2023, 13(22), 3462:
  * DenseNet-121 backbone, ImageNet weights, FC layers deleted
  * transition conv (1x1, S x S x 224) -> prediction conv (1x1, S x S x 14)
  * LogSumExp global pooling, weighted binary cross-entropy (W-CEL)
  * SGD momentum 0.9, weight decay 1e-4, lr 1e-3, lr /10 after epoch 20,
    batch 32, 100-epoch schedule (early stopping on val mean AUC)
  * 256 resize -> 224 random crop, +/-12 px translation, train-set normalisation

Dataset: /kaggle/input/jbeltranleon/nih-chest-xrays-224-rgb (224 px RGB PNGs).
Label CSVs are auto-discovered; if the mirror does not bundle them, attach the
official `nih-chest-xrays/data` dataset as a second input.

Usage (Kaggle notebook cell):
    !python lung_kaggle.py --data_root /kaggle/input/jbeltranleon/nih-chest-xrays-224-rgb \
        --output_dir /kaggle/working

Everything it writes (checkpoints, logs, plots, metrics, zip) lands in
--output_dir. Resume is automatic: rerun the same command after a session
timeout and training continues from `checkpoints/lung_last.pth`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import time
import zipfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Mixed-precision API moved from torch.cuda.amp to torch.amp in torch 2.4;
# support both so an older Kaggle image cannot crash on the first batch.
try:  # pragma: no cover - version shim
    from torch.amp import GradScaler as _GradScaler
    from torch.amp import autocast as _autocast
    _AMP_NEW_API = True
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore
    from torch.cuda.amp import autocast as _autocast  # type: ignore
    _AMP_NEW_API = False


def make_scaler(enabled: bool):
    if not enabled:
        return None
    return _GradScaler("cuda") if _AMP_NEW_API else _GradScaler()


def autocast_cuda(enabled: bool):
    return _autocast("cuda", enabled=enabled) if _AMP_NEW_API else _autocast(enabled=enabled)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lung_kaggle")

FINDINGS = (
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
)
FINDING_INDEX = {name: i for i, name in enumerate(FINDINGS)}
PAPER_AUC = {
    "Atelectasis": 0.82, "Cardiomegaly": 0.8725, "Effusion": 0.88,
    "Infiltration": 0.722, "Mass": 0.8406, "Pneumonia": 0.75,
    "Pneumothorax": 0.898, "Hernia": 0.76,
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_TAG = "znet_densenet121_224"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


def _patient_id(filename: str) -> str:
    return os.path.splitext(os.path.basename(filename))[0].split("_")[0]


def locate_root(root: str, max_depth: int = 3) -> str:
    """Find the directory that contains `images/`, breadth-first, with a listing on failure."""
    root = os.path.abspath(root)
    queue = [(root, 0)]
    visited = set()
    while queue:
        current, depth = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        if os.path.isdir(os.path.join(current, "images")):
            return current
        if depth >= max_depth:
            continue
        try:
            entries = sorted(os.listdir(current))
        except OSError:
            continue
        for entry in entries:
            path = os.path.join(current, entry)
            if os.path.isdir(path):
                queue.append((path, depth + 1))

    listing = []
    for base, depth in ((root, 1), (root, 2)):
        try:
            entries = sorted(os.listdir(base))
        except OSError:
            continue
        for entry in entries[:25]:
            path = os.path.join(base, entry)
            if os.path.isdir(path):
                listing.append(os.path.relpath(path, root))
    raise FileNotFoundError(
        f"No `images/` directory under {root!r} (searched {max_depth} levels).\n"
        f"Directories found: {listing or 'none — /kaggle/input looks empty, the dataset is not attached'}\n"
        "Attach the images dataset (Add Input → jbeltranleon/nih-chest-xrays-224-rgb) "
        "or pass --data_root /kaggle/input/<dataset-slug>."
    )


METADATA_NAMES = {
    "labels": ("Data_Entry_2017.csv", "Data_Entry_2017_v2020.csv"),
    "train_val": ("train_val_list.txt",),
    "test": ("test_list.txt",),
    "bbox": ("BBox_List_2017.csv",),
}


def _walk_dirs(base: str, max_depth: int = 4, cap: int = 20000) -> list[str]:
    """Bounded breadth-first directory walk; skips files entirely."""
    found: list[str] = []
    queue = [(base, 0)]
    seen: set[str] = set()
    while queue and len(found) < cap:
        current, depth = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        found.append(current)
        if depth >= max_depth:
            continue
        try:
            entries = sorted(os.listdir(current))
        except OSError:
            continue
        for entry in entries:
            path = os.path.join(current, entry)
            if os.path.isdir(path):
                queue.append((path, depth + 1))
    return found


def _search_roots(root: str, labels_dir: Optional[str]) -> list[str]:
    """Directories that may hold the NIH metadata files.

    Kaggle has moved between mount layouts — `/kaggle/input/<slug>` and
    `/kaggle/input/datasets/<owner>/<slug>` — so walk both the images root and
    its parent a few levels deep and check every directory.
    """
    roots: list[str] = []
    if labels_dir:
        roots.append(os.path.abspath(labels_dir))
    # Walk up to three ancestors — the label dataset may sit beside the images
    # dataset at any level (`/kaggle/input/<slug>`, `/kaggle/input/datasets/<owner>/<slug>`).
    ancestors = [root]
    cursor = root
    for _ in range(3):
        cursor = os.path.dirname(cursor.rstrip(os.sep))
        if not cursor or cursor == "/":
            break
        ancestors.append(cursor)
    roots.extend(ancestors)
    for base in ancestors:
        if base and os.path.isdir(base):
            roots.extend(_walk_dirs(base, max_depth=3))
    seen: set[str] = set()
    unique = []
    for path in roots:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def locate_metadata(root: str, kind: str, labels_dir: Optional[str] = None) -> Optional[str]:
    """Find a metadata file (labels / split lists / bboxes) across Kaggle inputs."""
    for directory in _search_roots(root, labels_dir):
        for name in METADATA_NAMES[kind]:
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return candidate
    return None


@dataclass(frozen=True)
class Sample:
    path: str
    labels: tuple
    patient: str


class ChestXray14(Dataset):
    def __init__(self, root: str, split: str, transform=None, seed: int = 42,
                 labels_dir: Optional[str] = None):
        self.root = locate_root(root)
        self.images_dir = os.path.join(self.root, "images")
        self.transform = transform
        self.samples: list[Sample] = []

        csv_path = locate_metadata(self.root, "labels", labels_dir)
        if csv_path is None:
            raise FileNotFoundError(
                "Data_Entry_2017.csv not found. The images dataset has no labels. "
                "Attach the official `nih-chest-xrays/data` dataset (or a small "
                "CSV-only dataset) as a second Kaggle input, or pass --labels_dir."
            )
        logger.info("Labels: %s", csv_path)
        labels_by_image: dict[str, tuple] = {}
        with open(csv_path, newline="") as handle:
            for row in csv.DictReader(handle):
                name = row.get("Image Index") or row.get("Image_Index") or ""
                text = row.get("Finding Labels") or row.get("Finding_Labels") or ""
                vector = [0.0] * len(FINDINGS)
                for finding in text.split("|"):
                    finding = finding.strip()
                    if finding in FINDING_INDEX:
                        vector[FINDING_INDEX[finding]] = 1.0
                if name:
                    labels_by_image[name] = tuple(vector)

        available = set(os.listdir(self.images_dir))
        labels_by_image = {k: v for k, v in labels_by_image.items() if k in available}

        train_val_path = locate_metadata(self.root, "train_val", labels_dir)
        test_path = locate_metadata(self.root, "test", labels_dir)
        if train_val_path and test_path:
            with open(train_val_path) as handle:
                train_val = {line.strip() for line in handle if line.strip()}
            with open(test_path) as handle:
                test = {line.strip() for line in handle if line.strip()}
            tv_patients = {_patient_id(n) for n in train_val if n in labels_by_image}
            test_patients = {_patient_id(n) for n in test if n in labels_by_image}
            tv_patients -= test_patients
            ordered = sorted(tv_patients)
            random.Random(seed).shuffle(ordered)
            n_val = int(round(0.125 * len(ordered)))
            val_patients = set(ordered[:n_val])
            train_patients = set(ordered[n_val:])
        else:
            logger.warning("Official split lists missing — deterministic 70/10/20 patient split.")
            patients = sorted({_patient_id(n) for n in labels_by_image})
            random.Random(seed).shuffle(patients)
            n_test, n_val = int(round(0.20 * len(patients))), int(round(0.10 * len(patients)))
            test_patients = set(patients[:n_test])
            val_patients = set(patients[n_test:n_test + n_val])
            train_patients = set(patients[n_test + n_val:])

        chosen = {"train": train_patients, "val": val_patients, "test": test_patients}[split]
        for name, vector in sorted(labels_by_image.items()):
            if _patient_id(name) in chosen:
                self.samples.append(Sample(os.path.join(self.images_dir, name), vector, _patient_id(name)))
        logger.info("%s split: %d images / %d patients", split, len(self.samples),
                    len({s.patient for s in self.samples}))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(sample.labels, dtype=torch.float32)


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomAffine(degrees=0, translate=(12 / 224, 12 / 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────


class ZNet(nn.Module):
    def __init__(self, num_classes: int = 14, pretrained: bool = True, transition_dim: int = 224):
        super().__init__()
        try:
            backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        except Exception as exc:  # offline Kaggle kernel without internet
            logger.warning("Pretrained weights unavailable (%s); training from scratch.", exc)
            backbone = densenet121(weights=None)
        self.features = backbone.features
        self.transition = nn.Conv2d(1024, transition_dim, 1)
        self.prediction = nn.Conv2d(transition_dim, num_classes, 1)
        for module in (self.transition, self.prediction):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(module.bias)

    def forward(self, x, return_maps: bool = False):
        fmap = F.relu(self.features(x))
        transition = F.relu(self.transition(fmap))
        class_maps = self.prediction(transition)
        logits = torch.logsumexp(class_maps.flatten(2), dim=2)
        if return_maps:
            return logits, class_maps
        return logits


class GradCAM:
    def __init__(self, model: ZNet):
        self.model = model
        self.activations = None
        self.gradients = None
        target = model.features.norm5
        self._handles = [
            target.register_forward_hook(self._save_activation),
            target.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.remove_hooks()

    def generate(self, tensor, class_index: int):
        if not tensor.requires_grad:
            tensor.requires_grad_(True)
        was_training = self.model.training
        self.model.eval()
        self.activations = None
        self.gradients = None
        try:
            logits = self.model(tensor)
            self.model.zero_grad(set_to_none=True)
            logits[0, int(class_index)].backward()
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * self.activations).sum(dim=1)).squeeze(0)
            cam = cam.detach().float().cpu().numpy()
            cam -= cam.min()
            if cam.max() > 0:
                cam /= cam.max()
            return cam
        finally:
            if was_training:
                self.model.train()
            self.model.zero_grad(set_to_none=True)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight(dataset: ChestXray14, clip: float = 50.0) -> torch.Tensor:
    positives = np.sum([s.labels for s in dataset.samples], axis=0, dtype=np.float64)
    negatives = len(dataset) - positives
    return torch.tensor(np.clip(negatives / np.clip(positives, 1.0, None), 1.0, clip),
                        dtype=torch.float32)


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running, total = 0.0, 0
    progress = tqdm(loader, desc=f"Epoch {epoch:02d} [train]", leave=False)
    for images, targets in progress:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_cuda(scaler is not None):
            logits = model(images)
            loss = criterion(logits, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += float(loss.item()) * images.size(0)
        total += images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return running / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, tag: str = "val"):
    model.eval()
    running, total = 0.0, 0
    probs, targets_all = [], []
    for images, targets in tqdm(loader, desc=f"[{tag}]", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets.to(device))
        running += float(loss.item()) * images.size(0)
        total += images.size(0)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        targets_all.append(targets.numpy())
    probs = np.vstack(probs)
    targets_all = np.vstack(targets_all)
    per_class = {}
    for index, name in enumerate(FINDINGS):
        column = targets_all[:, index]
        if column.min() == column.max():
            per_class[name] = float("nan")
        else:
            per_class[name] = float(roc_auc_score(column, probs[:, index]))
    valid = [v for v in per_class.values() if not math.isnan(v)]
    return running / max(total, 1), float(np.mean(valid)) if valid else 0.0, per_class, probs, targets_all


# ─────────────────────────────────────────────────────────────────────────────
# Artefacts
# ─────────────────────────────────────────────────────────────────────────────


def _save(figure, directory: str, name: str) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_curves(history: dict, out: str) -> str:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Weighted BCE loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["val_auc"], label="val mean AUC", color="tab:green")
    axes[1].axhline(0.858, ls="--", lw=1, color="tab:red", label="paper 0.858")
    axes[1].set_title("Mean AUC"); axes[1].set_xlabel("epoch"); axes[1].legend()
    return _save(figure, out, "training_curves.png")


def plot_per_class_auc(per_class: dict, out: str) -> str:
    names = [n for n in FINDINGS if not math.isnan(per_class.get(n, float("nan")))]
    values = [per_class[n] for n in names]
    figure, axis = plt.subplots(figsize=(12, 4.5))
    positions = np.arange(len(names))
    axis.bar(positions - 0.2, values, width=0.4, label="Z-Net (this run)")
    paper_names = [n for n in names if n in PAPER_AUC]
    axis.bar([names.index(n) + 0.2 for n in paper_names],
             [PAPER_AUC[n] for n in paper_names], width=0.4, label="paper")
    axis.set_xticks(positions); axis.set_xticklabels(names, rotation=35, ha="right")
    axis.set_ylim(0.5, 1.0); axis.axhline(np.mean(values), ls=":", color="gray")
    axis.set_title("Per-class test AUC vs paper"); axis.legend()
    return _save(figure, out, "per_class_auc.png")


def plot_roc(probs: np.ndarray, targets: np.ndarray, out: str) -> str:
    figure, axis = plt.subplots(figsize=(7, 6))
    for index, name in enumerate(FINDINGS):
        column = targets[:, index]
        if column.min() == column.max():
            continue
        fpr, tpr, _ = roc_curve(column, probs[:, index])
        axis.plot(fpr, tpr, lw=1, label=f"{name} ({roc_auc_score(column, probs[:, index]):.3f})")
    axis.plot([0, 1], [0, 1], "k--", lw=0.8)
    axis.set_xlabel("FPR"); axis.set_ylabel("TPR"); axis.set_title("Test ROC per finding")
    axis.legend(fontsize=6, loc="lower right")
    return _save(figure, out, "roc_curves.png")


def plot_class_distribution(dataset: ChestXray14, out: str) -> str:
    counts = np.sum([s.labels for s in dataset.samples], axis=0)
    figure, axis = plt.subplots(figsize=(12, 4.2))
    axis.bar(FINDINGS, counts)
    axis.set_ylabel("positive images"); axis.set_title("Training-set positives per finding")
    axis.tick_params(axis="x", rotation=35)
    return _save(figure, out, "class_distribution.png")


def plot_samples(model, dataset: ChestXray14, device, out: str, seed: int = 42) -> str:
    rng = random.Random(seed)
    picks = rng.sample(range(len(dataset)), min(8, len(dataset)))
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    model.eval()
    with torch.no_grad():
        for axis, index in zip(np.ravel(axes), picks):
            image, target = dataset[index]
            logits = model(image.unsqueeze(0).to(device))
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            top = int(np.argmax(probs))
            axis.imshow(_denormalise(image)); axis.axis("off")
            truth = [FINDINGS[i] for i, flag in enumerate(target.numpy()) if flag > 0.5] or ["No Finding"]
            axis.set_title(f"pred: {FINDINGS[top]} {probs[top]:.2f}\ntrue: {', '.join(truth)}", fontsize=8)
    return _save(figure, out, "sample_predictions.png")


def plot_cams(model, dataset: ChestXray14, device, out: str, seed: int = 42) -> str:
    rng = random.Random(seed)
    picks = rng.sample(range(len(dataset)), min(8, len(dataset)))
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    with GradCAM(model) as cam_gen:
        for axis, index in zip(np.ravel(axes), picks):
            image, target = dataset[index]
            tensor = image.unsqueeze(0).to(device)
            logits = model(tensor)
            top = int(torch.sigmoid(logits)[0].argmax())
            cam = cam_gen.generate(tensor, top)
            axis.imshow(_overlay(_denormalise(image), cam)); axis.axis("off")
            axis.set_title(f"Grad-CAM: {FINDINGS[top]}", fontsize=8)
    return _save(figure, out, "gradcam_samples.png")


def _denormalise(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    array = array * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(array, 0, 1)


def _overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    import cv2

    resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    if resized.max() <= 0:
        return image
    heat = cv2.applyColorMap(np.uint8(255 * resized / resized.max()), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip(alpha * heat + (1 - alpha) * image, 0, 1)


def evaluate_localization(model, root: str, device, out: str,
                          labels_dir: Optional[str] = None) -> Optional[dict]:
    """Paper Section 6.5: IoU between CAM boxes and the 984 annotated images."""
    csv_path = locate_metadata(locate_root(root), "bbox", labels_dir)
    if csv_path is None:
        logger.warning("BBox_List_2017.csv absent — skipping localisation IoU.")
        return None
    import cv2

    # The official header is `Bbox [x,y,w,h],,,` — unquoted, so csv splits the
    # box across columns. Parse by position and accept both the quoted and the
    # split forms.
    rows = []
    with open(csv_path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            name, label = row[0].strip(), row[1].strip()
            if row[2].strip().startswith("["):
                parts = row[2].strip().strip("[]").split(",")
            else:
                parts = row[2:6]
            try:
                x, y, w, h = [float(v) for v in parts[:4]]
            except (ValueError, IndexError):
                continue
            if name and label in FINDING_INDEX:
                rows.append((name, label, (x, y, w, h)))
    if not rows:
        return None

    images_dir = os.path.join(locate_root(root), "images")
    transform = build_transforms(train=False)
    hits, total = 0, 0
    per_label: dict[str, list[int]] = {}
    model.eval()
    with torch.no_grad():
        for name, label, (x, y, w, h) in tqdm(rows, desc="[localisation]", leave=False):
            path = os.path.join(images_dir, name)
            if not os.path.isfile(path):
                continue
            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            _, class_maps = model(tensor, return_maps=True)
            cam = class_maps[0, FINDING_INDEX[label]].float().cpu().numpy()
            cam = cv2.resize(cam, (224, 224))
            cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-6)
            mask = cam >= (60.0 / 255.0)
            if not mask.any():
                iou = 0.0
            else:
                ys, xs = np.where(mask)
                px0, py0, px1, py1 = xs.min(), ys.min(), xs.max(), ys.max()
                # BBox_List coordinates are in 1024-px space; scale to 224.
                gx0, gy0 = x * 224 / 1024, y * 224 / 1024
                gx1, gy1 = (x + w) * 224 / 1024, (y + h) * 224 / 1024
                inter = max(0, min(px1, gx1) - max(px0, gx0)) * max(0, min(py1, gy1) - max(py0, gy0))
                union = (px1 - px0) * (py1 - py0) + (gx1 - gx0) * (gy1 - gy0) - inter
                iou = inter / union if union > 0 else 0.0
            hit = int(iou >= 0.1)  # paper threshold T(IoU) = 0.1
            hits += hit
            total += 1
            per_label.setdefault(label, []).append(hit)
    if not total:
        return None
    result = {"iou_threshold": 0.1, "images": total, "accuracy": round(hits / total, 4),
              "per_label": {k: round(float(np.mean(v)), 4) for k, v in sorted(per_label.items())}}
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(list(result["per_label"].keys()), list(result["per_label"].values()))
    axis.set_ylim(0, 1); axis.set_title("Localisation accuracy (IoU >= 0.1)")
    axis.tick_params(axis="x", rotation=35)
    _save(figure, out, "localization_iou.png")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Z-Net on NIH ChestX-ray14")
    parser.add_argument("--data_root", default="/kaggle/input",
                        help="Images root, or /kaggle/input to auto-detect the images dataset")
    parser.add_argument("--labels_dir", default="",
                        help="Optional directory holding Data_Entry_2017.csv and the split lists")
    parser.add_argument("--output_dir", default="/kaggle/working")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transition_dim", type=int, default=224)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints = os.path.join(args.output_dir, "checkpoints")
    artifacts = os.path.join(args.output_dir, "artifacts")
    os.makedirs(checkpoints, exist_ok=True)
    os.makedirs(artifacts, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 70)
    logger.info("  Z-Net · NIH ChestX-ray14 · Ahmad et al. 2023 replication")
    logger.info("=" * 70)
    logger.info("  device=%s seed=%d batch=%d epochs=%d lr=%g wd=%g momentum=%.2f",
                device, args.seed, args.batch_size, args.epochs, args.lr,
                args.weight_decay, args.momentum)
    if torch.cuda.is_available():
        logger.info("  gpu=%s (%.1f GB)", torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    labels_dir = args.labels_dir or None
    train_set = ChestXray14(args.data_root, "train", build_transforms(True), seed=args.seed,
                            labels_dir=labels_dir)
    val_set = ChestXray14(args.data_root, "val", build_transforms(False), seed=args.seed,
                          labels_dir=labels_dir)
    test_set = ChestXray14(args.data_root, "test", build_transforms(False), seed=args.seed,
                           labels_dir=labels_dir)
    logger.info("Split sizes — train %d / val %d / test %d", len(train_set), len(val_set), len(test_set))

    def loader(dataset, shuffle):
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                          drop_last=shuffle, persistent_workers=args.num_workers > 0)

    train_loader = loader(train_set, True)
    val_loader = loader(val_set, False)
    test_loader = loader(test_set, False)

    raw_model = ZNet(len(FINDINGS), pretrained=True, transition_dim=args.transition_dim).to(device)
    model = raw_model
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(raw_model)
        logger.info("DataParallel across %d GPUs", torch.cuda.device_count())
    pos_weight = compute_pos_weight(train_set).to(device)
    logger.info("W-CEL pos_weight (neg/pos, clipped 50): %s", np.round(pos_weight.cpu().numpy(), 2))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    scaler = make_scaler(torch.cuda.is_available())

    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc, start_epoch = 0.0, 1
    last_path = os.path.join(checkpoints, "lung_last.pth")
    best_path = os.path.join(checkpoints, "lung_best_model.pth")
    if args.resume and os.path.isfile(last_path):
        state = torch.load(last_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state.get("history", history)
        best_auc = state.get("best_auc", 0.0)
        start_epoch = state.get("epoch", 0) + 1
        logger.info("Resumed from %s at epoch %d (best AUC %.4f)", last_path, start_epoch, best_auc)

    log_path = os.path.join(args.output_dir, "training_log.csv")
    write_header = not os.path.isfile(log_path) or start_epoch == 1
    log_handle = open(log_path, "a" if not write_header else "w", newline="")
    writer = csv.writer(log_handle)
    if write_header:
        writer.writerow(["epoch", "train_loss", "val_loss", "val_mean_auc", "lr", "seconds"])

    patience = 0
    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_auc, per_class, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()  # paper: lr / 10 after epoch 20 — epoch cadence, not batch
        elapsed = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_auc:.4f}",
                         f"{optimizer.param_groups[0]['lr']:.6f}", f"{elapsed:.1f}"])
        log_handle.flush()

        marker = ""
        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            torch.save(raw_model.state_dict(), best_path)
            marker = f" | best AUC={best_auc:.4f}"
        else:
            patience += 1
        logger.info("Epoch %02d/%02d | train %.4f | val %.4f AUC %.4f | %.0fs%s",
                    epoch, args.epochs, train_loss, val_loss, val_auc, elapsed, marker)

        torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "best_auc": best_auc, "history": history}, last_path)

        if patience >= args.patience:
            logger.info("Early stopping at epoch %d (patience %d on val mean AUC).", epoch, args.patience)
            break

    log_handle.close()
    logger.info("Training finished in %.1f minutes.", (time.time() - started) / 60)

    if not os.path.isfile(best_path):
        logger.error("No checkpoint saved — aborting evaluation.")
        return 1

    raw_model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_loss, test_auc, per_class, probs, targets = evaluate(raw_model, test_loader, criterion, device, "test")
    logger.info("TEST | loss %.4f | mean AUC %.4f", test_loss, test_auc)

    # Paper metrics: specificity at 0.5 and element-wise accuracy.
    predictions = (probs >= 0.5).astype(int)
    tn = ((predictions == 0) & (targets == 0)).sum(axis=0)
    fp = ((predictions == 1) & (targets == 0)).sum(axis=0)
    specificity = float(np.mean(tn / np.clip(tn + fp, 1, None)))
    element_acc = float((predictions == targets).mean())
    report = classification_report(targets.astype(int), predictions,
                                   target_names=list(FINDINGS), zero_division=0, digits=4)

    written = [
        plot_curves(history, artifacts),
        plot_per_class_auc(per_class, artifacts),
        plot_roc(probs, targets, artifacts),
        plot_class_distribution(train_set, artifacts),
        plot_samples(raw_model, test_set, device, artifacts, seed=args.seed),
        plot_cams(raw_model, test_set, device, artifacts, seed=args.seed),
    ]
    localization = evaluate_localization(raw_model, args.data_root, device, artifacts,
                                         labels_dir=labels_dir)

    metrics = {
        "model_tag": MODEL_TAG,
        "paper": "Ahmad et al., Diagnostics 2023, 13(22), 3462",
        "seed": args.seed,
        "epochs_run": len(history["train_loss"]),
        "best_val_mean_auc": round(best_auc, 6),
        "test_loss": round(test_loss, 6),
        "test_mean_auc": round(test_auc, 6),
        "test_specificity_at_0.5": round(specificity, 6),
        "test_elementwise_accuracy": round(element_acc, 6),
        "per_class_auc": {k: (None if math.isnan(v) else round(v, 6)) for k, v in per_class.items()},
        "paper_mean_auc": 0.858,
        "splits": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
        "hyperparameters": {
            "optimizer": "SGD", "momentum": args.momentum, "weight_decay": args.weight_decay,
            "lr": args.lr, "lr_step": "x0.1 @ epoch 20", "batch_size": args.batch_size,
            "loss": "weighted BCE (pos_weight = neg/pos, clip 50)",
            "augmentation": "resize 256, random crop 224, +/-12px translate",
        },
        "localization": localization,
    }
    with open(os.path.join(artifacts, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)
    with open(os.path.join(artifacts, "classification_report.txt"), "w") as handle:
        handle.write(f"Z-Net · NIH ChestX-ray14 · seed {args.seed}\n")
        handle.write(f"Test mean AUC {test_auc:.4f} (paper 0.858) · specificity {specificity:.4f}\n\n")
        handle.write(report)
    with open(os.path.join(artifacts, "split_manifest.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "patient", "split", "labels"])
        for name, dataset in (("train", train_set), ("val", val_set), ("test", test_set)):
            for sample in dataset.samples:
                labels = "|".join(FINDINGS[i] for i, f in enumerate(sample.labels) if f > 0.5)
                writer.writerow([os.path.basename(sample.path), sample.patient, name, labels or "No Finding"])

    zip_path = os.path.join(args.output_dir, "lung_classification_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, arcname in ((log_path, "training_log.csv"),
                              (best_path, "lung_best_model.pth"),
                              (last_path, "lung_last.pth")):
            if os.path.isfile(path):
                archive.write(path, arcname)
        for root, _, files in os.walk(artifacts):
            for name in files:
                full = os.path.join(root, name)
                archive.write(full, os.path.relpath(full, args.output_dir))

    logger.info("=" * 70)
    logger.info("  Run complete")
    logger.info("  checkpoint : %s", best_path)
    logger.info("  metrics    : %s", os.path.join(artifacts, "metrics.json"))
    logger.info("  zip        : %s", zip_path)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
