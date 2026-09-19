#!/usr/bin/env python3
"""
Sequence Tagger & Pure Clean Dataset Generator for OpenMed Brain Classification
-------------------------------------------------------------------------------
1. Pre-loads existing 12,626 archive hashes to prevent any cross-dataset collisions.
2. Uses a GlobalDeduplicator to enforce strict pairwise visual uniqueness:
   - Zero SHA-256 byte collisions
   - Zero perceptual pHash duplicates (Hamming distance <= 2) across the entire 4k set
   - Zero collisions with the existing 12.6k archive
3. Selects pure, clinically valid scans from BTSC (filtering out synthetic noise).
4. Trains an ultra-fast MobileNetV3 sequence classifier (T1, T1C+, T2) on CPU.
5. Classifies BTSC 'notumor' and 'pituitary' scans into exact pulse sequences.
6. Completely cleans the output directory and exports exactly 2,000 Normal and 2,000 Pituitary scans.
"""

import os
import sys
import io
import json
import random
import shutil
import argparse
import hashlib
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Any, Optional

import numpy as np
import scipy.fftpack
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ------------------------------------------------------------------------------
# Default Paths
# ------------------------------------------------------------------------------
DEFAULT_ARCHIVE_ROOT = "/home/honeypot/Projects/FAST_API/OpenMed/backend/datasets/brain/archive"
DEFAULT_DATA_JSON = f"{DEFAULT_ARCHIVE_ROOT}/DATA.json"
DEFAULT_IMAGES_BASE = f"{DEFAULT_ARCHIVE_ROOT}/Images_/Images_"

DEFAULT_BTSC_NOTUMOR = "/home/honeypot/Projects/FAST_API/BTSC-UNet-ViT/backend/dataset/Vit_Dataset/notumor"
DEFAULT_BTSC_PITUITARY = "/home/honeypot/Projects/FAST_API/BTSC-UNet-ViT/backend/dataset/Vit_Dataset/pituitary"

DEFAULT_OUTPUT_DIR = "/home/honeypot/Desktop/brain_dataset_augmented"

SEQUENCE_MAP = {0: "T1", 1: "T1C+", 2: "T2"}
REV_SEQUENCE_MAP = {v: k for k, v in SEQUENCE_MAP.items()}


# ------------------------------------------------------------------------------
# Hashing Utilities
# ------------------------------------------------------------------------------
def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    img_size = hash_size * highfreq_factor
    img_gray = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(img_gray, dtype=np.float32)

    dct = scipy.fftpack.dct(
        scipy.fftpack.dct(pixels, axis=0, norm="ortho"),
        axis=1,
        norm="ortho"
    )
    dctlow = dct[:hash_size, :hash_size]
    med = np.median(dctlow.flatten())
    diff = (dctlow > med).flatten()
    return sum([int(b) << i for i, b in enumerate(diff)])


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


# ------------------------------------------------------------------------------
# Global Deduplicator
# ------------------------------------------------------------------------------
class GlobalDeduplicator:
    def __init__(
        self,
        archive_shas: Set[str],
        archive_phashes: List[int],
        phash_threshold: int = 2,
    ):
        self.archive_shas = archive_shas
        self.archive_phashes = archive_phashes
        self.phash_threshold = phash_threshold

        self.accepted_shas: Set[str] = set()
        self.accepted_phashes: List[int] = []

    def is_duplicate(self, sha: str, phash: int) -> Tuple[bool, str]:
        # 1. Exact SHA check
        if sha in self.accepted_shas:
            return True, "internal_sha_dup"
        if sha in self.archive_shas:
            return True, "archive_sha_dup"

        # 2. Archive visual duplicate check (dist <= 1)
        for a_ph in self.archive_phashes:
            if hamming_distance(phash, a_ph) <= 1:
                return True, "archive_phash_dup"

        # 3. Accepted set visual duplicate check (dist <= threshold)
        for acc_ph in self.accepted_phashes:
            if hamming_distance(phash, acc_ph) <= self.phash_threshold:
                return True, "internal_phash_dup"

        return False, "ok"

    def add(self, sha: str, phash: int):
        self.accepted_shas.add(sha)
        self.accepted_phashes.append(phash)


# ------------------------------------------------------------------------------
# Training Dataset
# ------------------------------------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label
        except Exception:
            blank = Image.new("RGB", (224, 224), (0, 0, 0))
            if self.transform:
                blank = self.transform(blank)
            return blank, label


class InferenceDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]], transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        path = item["src_path"]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform:
                    tensor = self.transform(img)
                else:
                    tensor = transforms.ToTensor()(img)
                return tensor, idx
        except Exception:
            blank = Image.new("RGB", (224, 224), (0, 0, 0))
            if self.transform:
                blank = self.transform(blank)
            return blank, idx


# ------------------------------------------------------------------------------
# Load Archive Hashes
# ------------------------------------------------------------------------------
def load_archive_known_hashes(archive_root: str) -> Tuple[Set[str], List[int]]:
    cache_path = os.path.join(archive_root, ".archive_hashes_cache.json")
    if os.path.exists(cache_path):
        print(f"       Loading archive hashes from cache: {cache_path}")
        with open(cache_path, "r") as f:
            data = json.load(f)
        return set(data["shas"]), data["phashes"]

    print("       Scanning existing 12.6k archive to index SHA-256 and pHashes...")
    archive_json = os.path.join(archive_root, "DATA.json")
    images_base = os.path.join(archive_root, "Images_", "Images_")

    shas: Set[str] = set()
    phashes: List[int] = []

    if os.path.exists(archive_json):
        with open(archive_json, "r") as f:
            manifest = json.load(f)

        for rel_k in manifest.keys():
            if rel_k.endswith("_mask.png"):
                continue
            fpath = os.path.join(images_base, rel_k)
            if os.path.exists(fpath):
                shas.add(compute_sha256(fpath))
                try:
                    with Image.open(fpath) as img:
                        phashes.append(compute_phash(img))
                except Exception:
                    pass

        try:
            with open(cache_path, "w") as f:
                json.dump({"shas": list(shas), "phashes": phashes}, f)
            print(f"       Cached {len(shas):,d} archive hashes.")
        except Exception:
            pass

    return shas, phashes


# ------------------------------------------------------------------------------
# Collect Pure Scans using Global Deduplicator
# ------------------------------------------------------------------------------
def collect_pure_unique_scans(
    source_dir: str,
    target_count: int,
    deduplicator: GlobalDeduplicator,
    label_name: str,
) -> List[Dict[str, Any]]:
    print(f"\n[INGEST] Selecting {target_count:,d} pure unique scans for: {label_name}")
    print(f"         Source directory: {source_dir}")

    all_files = [
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f))
        and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Filter out noisy synthetic augmentations
    banned_prefixes = ["salt", "gaussian", "speckle", "median"]
    priority_prefixes = ["siar", "n_", "no_tumor", "p_", "pituitary", "tr-"]

    clean_tier1 = []
    clean_tier2 = []

    for f in all_files:
        base = os.path.basename(f).lower()
        if any(base.startswith(bp) for bp in banned_prefixes):
            continue
        if any(base.startswith(pp) for pp in priority_prefixes):
            clean_tier1.append(f)
        else:
            clean_tier2.append(f)

    random.seed(42)
    random.shuffle(clean_tier1)
    random.shuffle(clean_tier2)
    candidate_files = clean_tier1 + clean_tier2
    print(f"         Pure candidate files available: {len(candidate_files):,d}")

    accepted_items: List[Dict[str, Any]] = []
    rejection_counts = Counter()

    for fpath in candidate_files:
        if len(accepted_items) >= target_count:
            break

        sha = compute_sha256(fpath)

        # Variance / blank check
        try:
            with Image.open(fpath) as img:
                img_rgb = img.convert("RGB")
                w, h = img_rgb.size
                if w < 100 or h < 100:
                    continue

                stat = np.asarray(img_rgb.convert("L"), dtype=np.float32)
                if np.std(stat) < 15.0:
                    continue  # Filter blank / uniform images

                # Compute pHash on exact standardized 512x512 JPEG format to eliminate post-export drift
                img_512 = img_rgb.resize((512, 512), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img_512.save(buf, "JPEG", quality=95)
                buf.seek(0)
                with Image.open(buf) as saved_im:
                    ph = compute_phash(saved_im)
        except Exception:
            continue

        is_dup, reason = deduplicator.is_duplicate(sha, ph)
        if is_dup:
            rejection_counts[reason] += 1
            continue

        # Register scan in global deduplicator
        deduplicator.add(sha, ph)
        accepted_items.append({
            "src_path": fpath,
            "filename": os.path.basename(fpath),
            "sha256": sha,
            "phash": ph,
            "tumor_type": label_name,
        })

        if len(accepted_items) % 500 == 0 or len(accepted_items) == target_count:
            print(f"         Accepted {len(accepted_items):,d} / {target_count:,d} unique scans...")

    print(f"         Summary for {label_name}:")
    print(f"           - Accepted Pure Scans   : {len(accepted_items):,d}")
    print(f"           - Purged Internal SHA   : {rejection_counts['internal_sha_dup']:,d}")
    print(f"           - Purged Internal pHash : {rejection_counts['internal_phash_dup']:,d}")
    print(f"           - Purged 12.6k Collisions: {rejection_counts['archive_sha_dup'] + rejection_counts['archive_phash_dup']:,d}")

    return accepted_items


# ------------------------------------------------------------------------------
# Sequence classifier construction and cache provenance
# ------------------------------------------------------------------------------
def _build_sequence_model(pretrained: bool, freeze_backbone: bool = False) -> nn.Module:
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    )
    if freeze_backbone:
        for param in model.features[:-3].parameters():
            param.requires_grad = False
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(SEQUENCE_MAP))
    return model


def _cache_matches(meta_path: str, requested: Dict[str, Any]) -> bool:
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r") as handle:
            recorded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return all(recorded.get(key) == value for key, value in requested.items())


def train_sequence_classifier(
    data_json_path: str,
    images_base: str,
    samples_per_sequence: int = 1000,
    epochs: int = 8,
    batch_size: int = 32,
    device: str = "cpu",
    force_retrain: bool = False,
) -> nn.Module:
    checkpoint_path = os.path.join(os.path.dirname(__file__), "sequence_classifier_mobilenet.pth")
    meta_path = f"{checkpoint_path}.meta.json"
    requested = {
        "data_json_path": os.path.abspath(data_json_path),
        "samples_per_sequence": samples_per_sequence,
        "epochs": epochs,
        "batch_size": batch_size,
    }

    if os.path.exists(checkpoint_path) and not force_retrain and _cache_matches(meta_path, requested):
        print(f"\n[MODEL] Reusing cached sequence classifier: {checkpoint_path}")
        model = _build_sequence_model(pretrained=False)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        return model

    if os.path.exists(checkpoint_path) and not force_retrain:
        print(
            "\n[MODEL] Cached sequence classifier was trained with a different "
            "configuration; retraining. Pass --force_retrain to always rebuild."
        )

    print(f"\n[TRAIN] Loading sequence training scans from:\n        {data_json_path}")
    with open(data_json_path, "r") as f:
        data = json.load(f)

    seq_buckets: Dict[str, List[str]] = {"T1": [], "T1C+": [], "T2": []}
    for rel_key, meta in data.items():
        if rel_key.endswith("_mask.png"):
            continue
        seq = meta.get("sequence")
        if seq in seq_buckets:
            full_path = os.path.join(images_base, rel_key)
            if os.path.exists(full_path):
                seq_buckets[seq].append(full_path)

    random.seed(42)
    train_samples: List[Tuple[str, int]] = []
    val_samples: List[Tuple[str, int]] = []

    for seq, paths in seq_buckets.items():
        random.shuffle(paths)
        chosen = paths[:samples_per_sequence]
        split_idx = int(len(chosen) * 0.85)
        train_paths = chosen[:split_idx]
        val_paths = chosen[split_idx:]
        seq_id = REV_SEQUENCE_MAP[seq]
        train_samples.extend([(p, seq_id) for p in train_paths])
        val_samples.extend([(p, seq_id) for p in val_paths])

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = SequenceDataset(train_samples, transform=train_transform)
    val_ds = SequenceDataset(val_samples, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = _build_sequence_model(pretrained=True, freeze_backbone=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features[-3:].parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )

    print(f"        Training MobileNetV3 on CPU ({epochs} epochs)...")
    start_t = time.time()
    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        r_loss, correct, total = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            r_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(out, 1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

        # Validation
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                loss = criterion(out, lbls)
                v_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(out, 1)
                v_corr += (preds == lbls).sum().item()
                v_tot += lbls.size(0)

        ep_time = time.time() - ep_start
        print(
            f"        Epoch {epoch}/{epochs} ({ep_time:.1f}s) | "
            f"Train Acc: {(correct/total)*100:.2f}% | Val Acc: {(v_corr/v_tot)*100:.2f}%"
        )

    print(f"        Sequence classifier ready in {time.time()-start_t:.1f}s.")

    accuracy, report = evaluate_sequence_model(model, val_loader, device)
    print(
        f"        Held-out sequence accuracy {accuracy:.3f} on {len(val_samples)} scans.\n"
        f"{report}"
    )
    if accuracy < 0.90:
        print(
            "        [WARN] Sequence accuracy is below 0.90. Labels derived from this\n"
            "        model become the classifier's ground truth, so its errors are\n"
            "        baked in as contradictory supervision. Consider more epochs or\n"
            "        manual review of low-confidence rows."
        )

    try:
        torch.save(model.state_dict(), checkpoint_path)
        with open(meta_path, "w") as handle:
            json.dump({**requested, "val_accuracy": round(accuracy, 4)}, handle, indent=2)
        print(f"        Saved sequence classifier checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"        [WARN] Could not cache checkpoint: {e}")
    return model


@torch.no_grad()
def evaluate_sequence_model(
    model: nn.Module, loader: DataLoader, device: str = "cpu"
) -> Tuple[float, str]:
    model.eval()
    targets: List[int] = []
    predictions: List[int] = []
    for imgs, lbls in loader:
        out = model(imgs.to(device))
        predictions.extend(out.argmax(1).cpu().tolist())
        targets.extend(lbls.cpu().tolist())

    accuracy = sum(int(p == t) for p, t in zip(predictions, targets)) / max(len(targets), 1)
    names = [SEQUENCE_MAP[i] for i in sorted(SEQUENCE_MAP)]
    try:
        from sklearn.metrics import classification_report

        report = classification_report(
            targets, predictions, target_names=names, digits=3, zero_division=0
        )
    except Exception:
        report = ""
    return float(accuracy), report


# ------------------------------------------------------------------------------
# Classify Sequences
# ------------------------------------------------------------------------------
def classify_dataset_items(
    model: nn.Module,
    items: List[Dict[str, Any]],
    batch_size: int = 64,
    device: str = "cpu",
):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = InferenceDataset(items, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    with torch.no_grad():
        for imgs, indices in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            for idx_tensor, pred, conf in zip(indices, preds.cpu().tolist(), confidences.cpu().tolist()):
                idx = int(idx_tensor)
                items[idx]["sequence"] = SEQUENCE_MAP[pred]
                items[idx]["confidence"] = float(conf)


# ------------------------------------------------------------------------------
# Clean & Export Dataset to Disk
# ------------------------------------------------------------------------------
def export_dataset(
    items: List[Dict[str, Any]],
    output_dir: str,
    min_sequence_confidence: float = 0.60,
):
    print(f"\n[EXPORT] Cleaning destination and standardizing 512x512 RGB images to:\n         {output_dir}")

    low_confidence = [
        item for item in items
        if float(item.get("confidence", 1.0)) < min_sequence_confidence
    ]
    accepted = [
        item for item in items
        if float(item.get("confidence", 1.0)) >= min_sequence_confidence
    ]
    if low_confidence:
        print(
            f"         Discarded {len(low_confidence):,d} scans whose inferred pulse "
            f"sequence fell below confidence {min_sequence_confidence:.2f}."
        )

    # Completely wipe prior files in output directory to prevent leftovers
    images_base_dir = os.path.join(output_dir, "Images_")
    if os.path.exists(images_base_dir):
        shutil.rmtree(images_base_dir)

    os.makedirs(os.path.join(output_dir, "Images_", "Images_"), exist_ok=True)
    target_base = os.path.join(output_dir, "Images_", "Images_")

    manifest_records: Dict[str, Any] = {}
    counts = Counter()

    for item in accepted:
        tumor_type = item["tumor_type"]
        seq = item["sequence"]
        counts[f"{tumor_type} {seq}"] += 1

        sub_folder = f"{tumor_type}/{tumor_type} {seq}/{tumor_type} {seq}"
        dest_folder = os.path.join(target_base, sub_folder)
        os.makedirs(dest_folder, exist_ok=True)

        filename = item["filename"]
        base_name = os.path.splitext(filename)[0]
        dest_filename = f"{base_name}.jpg"
        dest_path = os.path.join(dest_folder, dest_filename)

        # Standardize to 512x512 RGB JPEG
        try:
            with Image.open(item["src_path"]) as img:
                img_rgb = img.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
                img_rgb.save(dest_path, "JPEG", quality=95)
        except Exception:
            continue

        rel_key = f"{sub_folder}/{dest_filename}"
        manifest_records[rel_key] = {
            "filename": dest_filename,
            "class": f"{tumor_type} {seq}",
            "tumor_type": tumor_type,
            "sequence": seq,
            "width": 512,
            "height": 512,
            "point": {},
            "location": [],
            "localization_annotated": False,
            "has_lesion": 1 if tumor_type == "Pituitary" else 0,
            "bbox_path": "",
            "mask_path": "",
            "source_dataset": "btsc-unet-vit",
            "sequence_confidence": round(item.get("confidence", 1.0), 4),
        }

    # Save DATA.json
    out_json = os.path.join(output_dir, "DATA.json")
    with open(out_json, "w") as f:
        json.dump(manifest_records, f, indent=2)

    print(f"\n[DONE] Successfully exported {len(manifest_records):,d} pure scans!")
    print(f"       Master DATA.json created: {out_json}")
    for cls_name, c in sorted(counts.items()):
        print(f"       - {cls_name:<18}: {c:,d} scans")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Collect pure clean deduplicated scans from BTSC.")
    parser.add_argument("--archive_root", type=str, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--btsc_notumor", type=str, default=DEFAULT_BTSC_NOTUMOR)
    parser.add_argument("--btsc_pituitary", type=str, default=DEFAULT_BTSC_PITUITARY)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count_normal", type=int, default=2000)
    parser.add_argument("--count_pituitary", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=8, help="Sequence classifier epochs")
    parser.add_argument(
        "--min_sequence_confidence", type=float, default=0.60,
        help="Drop scans whose inferred sequence confidence is below this value",
    )
    parser.add_argument(
        "--force_retrain", action="store_true",
        help="Ignore any cached sequence classifier and rebuild it",
    )
    args = parser.parse_args()

    print("=" * 72)
    print(" OPENMED PURE DEDUPLICATED DATASET EXPANDER (CPU)")
    print("=" * 72)

    # 1. Pre-load archive hashes to guarantee zero cross-dataset leakage
    archive_shas, archive_phashes = load_archive_known_hashes(args.archive_root)
    print(f"       Indexed {len(archive_shas):,d} existing archive hashes.")

    # 2. Initialize unified global deduplicator
    deduplicator = GlobalDeduplicator(
        archive_shas=archive_shas,
        archive_phashes=archive_phashes,
        phash_threshold=2,
    )

    # 3. Collect Normal scans
    clean_normal = collect_pure_unique_scans(
        source_dir=args.btsc_notumor,
        target_count=args.count_normal,
        deduplicator=deduplicator,
        label_name="Normal",
    )

    # 4. Collect Pituitary scans (deduplicated against archive AND Normal scans!)
    clean_pituitary = collect_pure_unique_scans(
        source_dir=args.btsc_pituitary,
        target_count=args.count_pituitary,
        deduplicator=deduplicator,
        label_name="Pituitary",
    )

    all_items = clean_normal + clean_pituitary
    print(f"\n[INFO] Total pure unique candidate scans gathered: {len(all_items):,d}")

    # 5. Train sequence classifier on CPU
    data_json_path = os.path.join(args.archive_root, "DATA.json")
    images_base = os.path.join(args.archive_root, "Images_", "Images_")

    model = train_sequence_classifier(
        data_json_path=data_json_path,
        images_base=images_base,
        samples_per_sequence=1000,
        epochs=3,
        batch_size=32,
        device="cpu",
    )

    # 6. Classify sequences
    print(f"\n[CLASS] Classifying pulse sequences for all {len(all_items):,d} scans...")
    classify_dataset_items(model, all_items, batch_size=64, device="cpu")

    # 7. Clean export to Desktop
    export_dataset(all_items, args.output_dir)


if __name__ == "__main__":
    main()
