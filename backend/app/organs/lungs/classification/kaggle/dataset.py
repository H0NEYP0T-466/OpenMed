"""Dataset, transforms and the group-aware stratified split.

Three things live here and they are all about not lying to yourself:

1. ``LungDataset`` reads an X-ray as grayscale, promotes it to three identical
   channels (the backbone expects 3), and applies a stochastic transform. The
   minority class draws the transform twice per sample (a 2x augmentation
   probability), not 2x copies on disk.
2. ``grouped_stratified_split`` splits by duplicate cluster, stratified by
   class. A cluster never spans two splits; that is asserted, not assumed.
3. ``compute_class_weights`` produces the inverse-frequency vector that the
   loss consumes, normalised to mean 1 so the effective learning rate does not
   move when the class mix changes.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from timm.data import create_transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Pillow >= 10.3 validates 1 <= bits <= 8 in ImageOps.posterize.
# timm's RandAugment with 'inc1' computes 4 - int((level / 10) * 4), which yields <= 0
# whenever level >= 10 (common with minority_rand_augment m11 or m9 + mstd0.5).
# Clamping bits to [1, 8] prevents ValueError in DataLoader worker processes.
_orig_imageops_posterize = ImageOps.posterize


def _safe_imageops_posterize(image: Image.Image, bits: int) -> Image.Image:
    return _orig_imageops_posterize(image, max(1, min(8, int(bits))))


ImageOps.posterize = _safe_imageops_posterize

try:
    from timm.data import auto_augment

    _orig_timm_posterize = auto_augment.posterize

    def _safe_timm_posterize(img: Image.Image, bits_to_keep: int, **kwargs: Any) -> Image.Image:
        return _orig_timm_posterize(img, max(1, min(8, int(bits_to_keep))), **kwargs)

    auto_augment.posterize = _safe_timm_posterize
    auto_augment.NAME_TO_OP["Posterize"] = _safe_timm_posterize
    auto_augment.NAME_TO_OP["PosterizeIncreasing"] = _safe_timm_posterize
    auto_augment.NAME_TO_OP["PosterizeOriginal"] = _safe_timm_posterize

    _orig_post_inc = auto_augment._posterize_increasing_level_to_arg

    def _safe_post_inc(level: float, hparams: Any) -> tuple[int]:
        res = _orig_post_inc(level, hparams)
        return (max(1, min(8, int(res[0]))),)

    auto_augment._posterize_increasing_level_to_arg = _safe_post_inc
    auto_augment.LEVEL_TO_ARG["PosterizeIncreasing"] = _safe_post_inc
except Exception:
    pass

try:  # package import: app.organs.lungs.classification.kaggle.dataset
    from . import hygiene
    from .config import IMAGENET_MEAN, IMAGENET_STD, LungConfig
except ImportError:  # direct execution: python train_lungs_v2s.py
    import hygiene
    from config import IMAGENET_MEAN, IMAGENET_STD, LungConfig

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


# ─────────────────────────────────────────────────────────────────────────────
# Image IO and transforms
# ─────────────────────────────────────────────────────────────────────────────


def load_rgb(path: str | os.PathLike[str]) -> Image.Image:
    """Open an X-ray as 3-channel RGB.

    ``convert("L")`` first is deliberate: a PNG that happens to be RGB (or
    RGBA) is reduced to its luminance, so the model never sees a colour cue that
    does not exist in radiography, and the three channels are identical.
    """
    with Image.open(path) as image:
        return image.convert("L").convert("RGB")


def build_train_transform(
    size: int,
    cfg: LungConfig,
    rand_augment: Optional[str] = None,
) -> transforms.Compose:
    """RandAugment + RRC(0.8-1.0) + hflip + +/-10 deg rotation, colour jitter off.

    timm's ``create_transform`` is used for the augmentation block because it
    implements the full ``rand-m9-mstd0.5-inc1`` spec (magnitude 9, magnitude
    std 0.5, increasing magnitude), which torchvision's RandAugment cannot
    express. Rotation is prepended on the PIL image.

    ``rand_augment`` overrides the configured policy, which is how the minority
    class gets a genuinely stronger transform (``rand-m11``) instead of a second
    independent draw of the same one — two draws of one distribution keep the
    same distribution, so a second draw alone augments nothing.
    """
    augment = create_transform(
        input_size=size,
        is_training=True,
        auto_augment=rand_augment or cfg.rand_augment,
        re_prob=0.0,
        hflip=cfg.hflip_p,
        vflip=0.0,
        color_jitter=None,
        scale=cfg.rrc_scale,
        ratio=cfg.rrc_ratio,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        interpolation="bicubic",
    )
    return transforms.Compose(
        [transforms.RandomRotation(degrees=cfg.rotation_deg, fill=0), augment]
    )


def build_eval_transform(size: int) -> transforms.Compose:
    """Deterministic resize + ImageNet normalisation.

    Deliberately a bare bicubic resize to (size, size) with no centre crop, so
    this is byte-identical to ``preprocess_v2s`` in ``model_v2s.py``. timm's
    ``create_transform(is_training=False)`` resizes to size/crop_pct and then
    centre-crops, which frames the image differently from serving — validation
    would then measure a transform the user never runs.
    """
    return transforms.Compose(
        [
            transforms.Resize(
                (size, size), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation for plotting. Returns HxWx3 uint8."""
    array = tensor.detach().cpu().float().numpy()
    if array.ndim != 3:
        raise ValueError(f"expected CHW tensor, got shape {array.shape}")
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    array = array * std + mean
    array = np.clip(array, 0.0, 1.0).transpose(1, 2, 0)
    return (array * 255.0).astype(np.uint8)


def preview_augmentations(
    records: Sequence[hygiene.ImageRecord],
    index: int,
    transform: Any,
    count: int = 8,
    seed: int = 42,
) -> list[tuple[str, np.ndarray]]:
    """``count`` independent draws of the training transform for one image."""
    random.seed(seed)
    image = load_rgb(records[index].path)
    panels: list[tuple[str, np.ndarray]] = []
    for draw in range(count):
        panels.append((f"draw {draw + 1}", denormalise(transform(image))))
    return panels


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


class LungDataset(Dataset):
    """A split of the corpus, addressed by indices into the record list.

    ``__getitem__`` returns ``(tensor, class_index, record_index)``. The record
    index travels with the batch so a manifest or a Grad-CAM can name the exact
    file without a second lookup table.
    """

    def __init__(
        self,
        records: Sequence[hygiene.ImageRecord],
        indices: Sequence[int],
        transform: Any,
        minority_class_index: Optional[int] = None,
        minority_extra_draw_prob: float = 0.0,
        minority_transform: Any = None,
    ) -> None:
        self.records = records
        self.indices = list(indices)
        self.transform = transform
        self.minority_class_index = minority_class_index
        self.minority_extra_draw_prob = float(minority_extra_draw_prob)
        self.minority_transform = minority_transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int, int]:
        record_index = self.indices[position]
        record = self.records[record_index]
        image = load_rgb(record.path)
        use_minority_policy = (
            self.minority_transform is not None
            and self.minority_class_index is not None
            and record.class_index == self.minority_class_index
            and self.minority_extra_draw_prob > 0.0
            and random.random() < self.minority_extra_draw_prob
        )
        # A *stronger* policy (rand-m11), not a second draw of the same one: two
        # draws from one distribution are still that distribution, so a repeat
        # alone changes nothing. Branching before the call also avoids paying for
        # a transform whose output would be discarded.
        transform = self.minority_transform if use_minority_policy else self.transform
        tensor = transform(image)
        return tensor, int(record.class_index), int(record_index)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int = 42,
    drop_last: Optional[bool] = None,
) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(shuffle) if drop_last is None else drop_last,
        generator=generator,
        persistent_workers=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Group-aware stratified split
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SplitPlan:
    """Result of the split: indices per split plus the bookkeeping to audit it."""

    train: list[int] = field(default_factory=list)
    val: list[int] = field(default_factory=list)
    test: list[int] = field(default_factory=list)
    split_of: dict[int, str] = field(default_factory=dict)
    cluster_class: dict[int, int] = field(default_factory=dict)
    per_class: dict[str, dict[str, int]] = field(default_factory=dict)

    def indices(self, split: str) -> list[int]:
        if split == "train":
            return self.train
        if split == "val":
            return self.val
        if split == "test":
            return self.test
        raise KeyError(f"unknown split {split!r}")

    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _cluster_labels(
    records: Sequence[hygiene.ImageRecord], cluster_members: dict[int, list[int]]
) -> dict[int, int]:
    """Dominant class of each cluster (clusters can straddle two labels)."""
    cluster_class: dict[int, int] = {}
    for cluster_id, members in cluster_members.items():
        counts = np.bincount(
            [records[index].class_index for index in members], minlength=1
        )
        cluster_class[cluster_id] = int(counts.argmax())
    return cluster_class


def grouped_stratified_split(
    records: Sequence[hygiene.ImageRecord],
    cluster_members: dict[int, list[int]],
    class_names: Sequence[str],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> SplitPlan:
    """Split by duplicate cluster, stratified per class, 70/15/15 by default.

    Clusters are assigned largest-first to whichever split is furthest below its
    target share of that class. Because assignment happens at cluster level, a
    cluster can never straddle two splits; :func:`assert_cluster_isolation`
    verifies that rather than trusting the loop.
    """
    if not (0.0 < val_frac < 1.0) or not (0.0 < test_frac < 1.0):
        raise ValueError("val_frac and test_frac must be in (0, 1)")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must leave a training split")
    if not cluster_members:
        raise RuntimeError("No clusters to split; the hygiene scan produced nothing.")

    train_frac = 1.0 - val_frac - test_frac
    fractions = {"train": train_frac, "val": val_frac, "test": test_frac}
    cluster_class = _cluster_labels(records, cluster_members)

    rng = random.Random(seed)
    split_of: dict[int, str] = {}

    for class_index, class_name in enumerate(class_names):
        cluster_ids = [
            cluster_id
            for cluster_id in cluster_members
            if cluster_class[cluster_id] == class_index
        ]
        if len(cluster_ids) < 3:
            raise RuntimeError(
                f"Class {class_name!r} has only {len(cluster_ids)} duplicate cluster(s); "
                "a leak-free 70/15/15 split needs at least 3. The dataset folder is "
                "probably wrong or mostly duplicates."
            )

        sizes = {cluster_id: len(cluster_members[cluster_id]) for cluster_id in cluster_ids}
        total = float(sum(sizes.values()))
        targets = {split: total * fractions[split] for split in SPLIT_NAMES}

        order = list(cluster_ids)
        rng.shuffle(order)
        order.sort(key=lambda cluster_id: -sizes[cluster_id])

        current = dict.fromkeys(SPLIT_NAMES, 0.0)
        for cluster_id in order:
            deficit = {
                split: (targets[split] - current[split]) / max(targets[split], 1.0)
                for split in SPLIT_NAMES
            }
            chosen = max(SPLIT_NAMES, key=lambda split: (deficit[split], split == "train"))
            for member in cluster_members[cluster_id]:
                split_of[member] = chosen
            current[chosen] += sizes[cluster_id]

    # Count by the label each image actually carries: a cross-class cluster is
    # assigned as a whole, so one of its images lands under the other class.
    per_class = {name: dict.fromkeys(SPLIT_NAMES, 0) for name in class_names}
    for index, split in split_of.items():
        per_class[class_names[records[index].class_index]][split] += 1

    for class_name, counts in per_class.items():
        empty = [split for split, count in counts.items() if count == 0]
        if empty:
            raise RuntimeError(
                f"Class {class_name!r} ended up empty in split(s) {empty}. "
                "Too few clusters to stratify; merge small clusters or lower val/test."
            )

    plan = SplitPlan(
        split_of=split_of, cluster_class=cluster_class, per_class=per_class
    )
    for index, split in split_of.items():
        plan.indices(split).append(index)
    for split in SPLIT_NAMES:
        plan.indices(split).sort()

    assert_cluster_isolation(records, cluster_members, plan.split_of)
    return plan


def assert_cluster_isolation(
    records: Sequence[hygiene.ImageRecord],
    cluster_members: dict[int, list[int]],
    split_of: dict[int, str],
) -> None:
    """Fail loudly if any duplicate cluster touches more than one split."""
    offenders: list[str] = []
    for cluster_id, members in cluster_members.items():
        splits = {split_of[index] for index in members if index in split_of}
        if len(splits) > 1:
            sample = ", ".join(records[index].rel_path for index in members[:3])
            offenders.append(f"cluster {cluster_id} -> {sorted(splits)} ({sample})")
    if offenders:
        raise RuntimeError(
            "Leakage: duplicate cluster(s) span more than one split. "
            "This must never happen; the split is invalid.\n  "
            + "\n  ".join(offenders[:20])
        )


def compute_class_weights(
    train_indices: Sequence[int],
    records: Sequence[hygiene.ImageRecord],
    num_classes: int,
) -> np.ndarray:
    """Inverse-frequency weights, normalised so their mean is exactly 1."""
    if not train_indices:
        raise ValueError("train_indices is empty; cannot compute class weights")
    counts = np.zeros(num_classes, dtype=np.float64)
    for index in train_indices:
        counts[records[index].class_index] += 1.0
    if (counts == 0).any():
        missing = [int(i) for i in np.where(counts == 0)[0]]
        raise RuntimeError(f"Class index(es) {missing} absent from the training split")
    weights = 1.0 / counts
    return weights / weights.mean()


def class_counts(
    indices: Iterable[int],
    records: Sequence[hygiene.ImageRecord],
    num_classes: int,
) -> list[int]:
    counts = [0] * num_classes
    for index in indices:
        counts[records[index].class_index] += 1
    return counts


def split_counts(
    plan: SplitPlan,
    records: Sequence[hygiene.ImageRecord],
    class_names: Sequence[str],
) -> dict[str, dict[str, int]]:
    """``{split: {class: count}}`` for the distribution plot and the log."""
    out: dict[str, dict[str, int]] = {}
    for split in SPLIT_NAMES:
        counts = class_counts(plan.indices(split), records, len(class_names))
        out[split] = {name: counts[i] for i, name in enumerate(class_names)}
    return out


def manifest_rows(
    records: Sequence[hygiene.ImageRecord], plan: SplitPlan
) -> list[list[Any]]:
    """Rows for ``split_manifest.csv``: path, class, sha256, phash, cluster, split."""
    rows: list[list[Any]] = []
    for split in SPLIT_NAMES:
        for index in plan.indices(split):
            record = records[index]
            rows.append(
                [
                    record.rel_path,
                    record.class_name,
                    record.sha256 or "",
                    record.phash_hex or "",
                    record.cluster_id,
                    split,
                ]
            )
    return rows


__all__ = [
    "LungDataset",
    "SplitPlan",
    "assert_cluster_isolation",
    "build_eval_transform",
    "build_train_transform",
    "class_counts",
    "compute_class_weights",
    "denormalise",
    "grouped_stratified_split",
    "load_rgb",
    "make_loader",
    "manifest_rows",
    "preview_augmentations",
    "split_counts",
]
