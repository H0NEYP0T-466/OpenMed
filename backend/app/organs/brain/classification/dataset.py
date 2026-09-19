import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class BrainTumorDataset(Dataset):
    """
    Dataset for Brain Tumor Classification.
    Loads DATA.json and filters out corrupt mask images.
    """
    CLASS_NAMES = []
    TUMOR_TYPES = []
    @staticmethod
    def locate_data_and_images(data_root):
        """Intelligently discovers DATA.json and images base in any layout (nested, zipped, Kaggle)."""
        json_path = None
        data_dir = data_root

        if os.path.isfile(data_root):
            if data_root.endswith('.json'):
                json_path = data_root
                data_dir = os.path.dirname(data_root)
            elif data_root.endswith('.zip'):
                import zipfile
                extract_target = os.path.join(os.path.dirname(data_root), "unzipped_dataset")
                if not os.path.isdir(extract_target):
                    logger.info(f"Extracting {data_root} to {extract_target}...")
                    with zipfile.ZipFile(data_root, 'r') as z:
                        z.extractall(extract_target)
                data_dir = extract_target

        if json_path is None:
            candidates = [
                os.path.join(data_dir, "DATA.json"),
                os.path.join(data_dir, "archive", "DATA.json"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    json_path = c
                    data_dir = os.path.dirname(c)
                    break

        if json_path is None and os.path.isdir(data_root):
            for root, _, files in os.walk(data_root):
                if "DATA.json" in files:
                    json_path = os.path.join(root, "DATA.json")
                    data_dir = root
                    break

        if json_path is None:
            raise FileNotFoundError(f"Could not locate DATA.json inside '{data_root}'. Please verify the path.")

        img_candidates = [
            os.path.join(data_dir, "Images_", "Images_"),
            os.path.join(data_dir, "Images_"),
            os.path.join(os.path.dirname(data_dir), "Images_", "Images_"),
            os.path.join(os.path.dirname(data_dir), "Images_"),
            data_dir,
        ]
        images_base = data_dir
        for c in img_candidates:
            if os.path.isdir(c):
                images_base = c
                break

        return json_path, images_base

    def __init__(self, data_root, transform=None, split_indices=None):
        """
        Args:
            data_root (str): Path to dataset archive, DATA.json, or parent directory on Kaggle.
            transform (callable, optional): Optional transform to be applied on a sample.
            split_indices (list, optional): List of indices to subset the dataset (for train/val/test).
        """
        self.json_path, self.images_base = self.locate_data_and_images(data_root)
        self.data_root = os.path.dirname(self.json_path)
        self.transform = transform
        
        with open(self.json_path, 'r') as f:
            raw_data = json.load(f)
            
        self.samples = []
        classes_set = set()
        tumor_types_set = set()
        
        for relative_path, metadata in raw_data.items():
            if relative_path.endswith('_mask.png'):
                continue
                
            classes_set.add(metadata['class'])
            tumor_types_set.add(metadata['tumor_type'])
            
            self.samples.append({
                'path': os.path.join(self.images_base, relative_path),
                'relative_path': relative_path,
                'metadata': metadata
            })
            
        # Sort samples by relative_path for reproducibility before applying split
        self.samples.sort(key=lambda x: x['relative_path'])
            
        if not BrainTumorDataset.CLASS_NAMES:
            BrainTumorDataset.CLASS_NAMES = sorted(list(classes_set))
            BrainTumorDataset.TUMOR_TYPES = sorted(list(tumor_types_set))
            
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(BrainTumorDataset.CLASS_NAMES)}
        
        if split_indices is not None:
            self.samples = [self.samples[i] for i in split_indices]
            
        logger.info(f"Loaded dataset with {len(self.samples)} samples. ({len(classes_set)} classes)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['path']
        metadata = sample['metadata']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (512, 512), (0, 0, 0))
            
        label = self.class_to_idx[metadata['class']]
        
        if self.transform:
            image = self.transform(image)
            
        meta_dict = {
            'location': metadata.get('location', []),
            'point': metadata.get('point', []),
            'sequence': metadata.get('sequence', ''),
            'tumor_type': metadata.get('tumor_type', ''),
            'filename': metadata.get('filename', ''),
            'has_lesion': metadata.get('has_lesion', 0)
        }
            
        return image, label, meta_dict

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle variable-length metadata lists."""
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        metas = [item[2] for item in batch]
        return images, labels, metas
        
    @staticmethod
    def compute_class_weights(dataset_samples, num_classes):
        """
        Computes robust smoothed inverse-frequency class weights for cross-entropy loss.
        Applies square-root dampening and clipping to prevent minority/majority class explosion.
        """
        class_counts = Counter([sample['metadata']['class'] for sample in dataset_samples])
        class_names = BrainTumorDataset.CLASS_NAMES
        
        weights = np.zeros(num_classes, dtype=np.float32)
        total_samples = len(dataset_samples)
        
        for i, cls_name in enumerate(class_names):
            count = class_counts.get(cls_name, 0)
            if count > 0:
                # Square-root smoothed inverse frequency
                weights[i] = np.sqrt(total_samples / (num_classes * count))
            else:
                weights[i] = 1.0  # fallback
                
        # Normalize weights so mean is 1.0
        mean_w = np.mean(weights)
        if mean_w > 0:
            weights = weights / mean_w
            
        # Clip extreme weights to avoid gradient instability
        weights = np.clip(weights, 0.2, 5.0)
        
        return torch.tensor(weights, dtype=torch.float)

    @staticmethod
    def get_stratified_splits(data_root, test_size=0.15, val_size=0.15, seed=42):
        """
        Returns train, val, test index lists.
        Stratifies by primary location to prevent anatomy leakage, 
        falling back to class stratification.
        """
        json_path, _ = BrainTumorDataset.locate_data_and_images(data_root)
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
            
        samples = []
        for relative_path, metadata in raw_data.items():
            if relative_path.endswith('_mask.png'):
                continue
            samples.append({
                'relative_path': relative_path,
                'metadata': metadata
            })
            
        samples.sort(key=lambda x: x['relative_path'])
        
        # Create stratification labels: Primary location + Class
        stratify_labels = []
        for s in samples:
            locations = s['metadata'].get('location', [])
            primary_loc = locations[0] if locations else "Unknown"
            cls = s['metadata']['class']
            stratify_labels.append(f"{primary_loc}_{cls}")
            
        # Fallback to just class if combination is too rare for splitting
        counts = Counter(stratify_labels)
        safe_labels = [label if counts[label] >= 3 else s['metadata']['class'] for label, s in zip(stratify_labels, samples)]
        
        indices = np.arange(len(samples))
        
        val_test_ratio = test_size + val_size
        
        # First split: train vs (val + test)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=seed)
        try:
            train_idx, val_test_idx = next(sss.split(indices, safe_labels))
        except ValueError:
            # Fallback to class only if still failing
            classes = [s['metadata']['class'] for s in samples]
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=seed)
            train_idx, val_test_idx = next(sss.split(indices, classes))
            
        # Second split: val vs test
        val_test_safe_labels = [safe_labels[i] for i in val_test_idx]
        test_ratio_of_remainder = test_size / val_test_ratio
        
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio_of_remainder, random_state=seed)
        try:
            val_idx_rel, test_idx_rel = next(sss2.split(val_test_idx, val_test_safe_labels))
        except ValueError:
            classes_val_test = [samples[i]['metadata']['class'] for i in val_test_idx]
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio_of_remainder, random_state=seed)
            val_idx_rel, test_idx_rel = next(sss2.split(val_test_idx, classes_val_test))
            
        val_idx = val_test_idx[val_idx_rel]
        test_idx = val_test_idx[test_idx_rel]
        
        logger.info(f"Splits created: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        
        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()
