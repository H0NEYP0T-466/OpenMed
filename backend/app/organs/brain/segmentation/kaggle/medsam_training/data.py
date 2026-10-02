"""Stage 3 — split construction, dataset and transforms.

Split policy
------------
Two sources, two different grouping realities:

**OpenMed** — filenames encode lesion and slice (``T1C+ - Cystic glioblastoma
occipital , ventricle 005``). Stripping the sequence prefix and the trailing
slice number yields a **lesion key**, so slices of one lesion can be held
together. This is the same key the earlier cleaning pass used.

**BTSC** — filenames are ``1.png … 3064.png``. The figshare source *does* carry
a patient id (``cjdata/pid`` inside the original ``.mat`` files) but the PNG
re-export drops it, and the Kaggle copy contains no metadata file. So there is
no patient key to group on. Random splitting would separate adjacent slices of
one scan; instead the index ordering is cut into **contiguous blocks**, which
keeps each run of neighbouring slices on one side of the split.

Blocks are an improvement, not a proof. Each block boundary is a place where a
scan could straddle two splits, and the number of slices exposed that way is
measured and reported rather than assumed away. The manifest carries
``group_is_subject_verified: false`` for BTSC so nothing downstream can
mistake the block key for a patient key.

Both sources appear in all three splits, which is what the run needs in order
to report an in-domain number and a cross-source number separately.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .audit import SourceSpec, list_stems, resolve_image_path, resolve_mask_path
from .preprocessor import MEDSAM_INPUT_SIZE, preprocess_image

logger = logging.getLogger("data")

# Sequence prefixes seen in the OpenMed filenames. Kept identical to the
# pattern used by the cleaning scripts so the group keys match.
_SEQ_PREFIX = re.compile(r"^(T1C\+|T1|T2)\s*-\s*")
_TRAILING_NUMBER = re.compile(r"\s+\d+$")


def openmed_group_key(stem: str) -> str:
    """Lesion key for an OpenMed filename. Not a patient key — see module docstring."""
    return _SEQ_PREFIX.sub("", _TRAILING_NUMBER.sub("", stem))


# ─────────────────────────────────────────────────────────────────────────────
# Records
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Record:
    source: str
    stem: str
    image_path: str
    mask_path: str
    group: str
    group_is_subject_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "stem": self.stem,
            "image": self.image_path,
            "mask": self.mask_path,
            "group": self.group,
            "group_is_subject_verified": self.group_is_subject_verified,
        }


def build_records(spec: SourceSpec) -> list[Record]:
    """Resolve every paired stem into a Record, assigning group keys.

    When the source has no group callable, the group is filled in later by the
    block splitter — ``group`` is left as the stem until then.
    """
    records: list[Record] = []
    for stem in list_stems(spec):
        group = str(spec.group_of(stem)) if spec.group_of is not None else stem
        records.append(
            Record(
                source=spec.name,
                stem=stem,
                image_path=resolve_image_path(spec, stem),
                mask_path=resolve_mask_path(spec, stem),
                group=group,
                group_is_subject_verified=spec.group_is_subject_verified,
            )
        )
    logger.info("[%s] %d records", spec.name, len(records))
    return records


def assign_blocks(records: list[Record], block_size: int) -> list[Record]:
    """Give block-based group keys to a source that has none.

    Slices are ordered by their numeric stem when the stems are integers (BTSC
    is), otherwise lexicographically. Consecutive runs of ``block_size`` become
    one group. This does not recover scans; it bounds how far a split can cut
    into a run of neighbouring slices.
    """
    def sort_key(rec: Record):
        try:
            return (0, int(rec.stem), rec.stem)
        except ValueError:
            return (1, 0, rec.stem)

    ordered = sorted(records, key=sort_key)
    for index, rec in enumerate(ordered):
        rec.group = f"block_{index // block_size:05d}"
    logger.info(
        "[%s] assigned %d contiguous blocks of %d slices (boundaries=%d)",
        ordered[0].source if ordered else "?", len({r.group for r in ordered}),
        block_size, max(0, len({r.group for r in ordered}) - 1),
    )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Splitting
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SplitManifest:
    train: list[Record] = field(default_factory=list)
    val: list[Record] = field(default_factory=list)
    test: list[Record] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)

    def all_records(self) -> list[Record]:
        return [*self.train, *self.val, *self.test]

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        for name, part in (("train", self.train), ("val", self.val), ("test", self.test)):
            with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
                json.dump([r.to_dict() for r in part], fh, indent=2)

    @classmethod
    def load(cls, out_dir: str) -> "SplitManifest":
        manifest = cls()
        for name in ("train", "val", "test"):
            with open(os.path.join(out_dir, f"{name}.json")) as fh:
                payload = json.load(fh)
            setattr(
                manifest, name,
                [
                    Record(
                        source=row["source"], stem=row["stem"],
                        image_path=row["image"], mask_path=row["mask"],
                        group=row["group"],
                        group_is_subject_verified=row.get("group_is_subject_verified", False),
                    )
                    for row in payload
                ],
            )
        return manifest


def _partition_groups(
    records: list[Record], val_frac: float, test_frac: float, rng: random.Random
) -> tuple[list[Record], list[Record], list[Record]]:
    """Greedy group-level split hitting the target fractions by slice count.

    Groups are shuffled, then walked once. A group goes to val until val has
    enough slices, then to test until test has enough, then to train. Because
    allocation is per group, the achieved fractions land close to the target
    without ever splitting a group.
    """
    groups: dict[str, list[Record]] = defaultdict(list)
    for rec in records:
        groups[rec.group].append(rec)

    keys = sorted(groups)
    rng.shuffle(keys)

    total = len(records)
    val_target = int(total * val_frac)
    test_target = int(total * test_frac)

    val: list[Record] = []
    test: list[Record] = []
    train: list[Record] = []

    for key in keys:
        bucket = groups[key]
        if len(val) < val_target:
            val.extend(bucket)
        elif len(test) < test_target:
            test.extend(bucket)
        else:
            train.extend(bucket)
    return train, val, test


def build_splits(
    openmed: SourceSpec,
    btsc: SourceSpec,
    *,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    btsc_block_size: int = 32,
    seed: int = 42,
) -> SplitManifest:
    """Build the combined split and verify that no group crosses a boundary."""
    rng = random.Random(seed)

    om_records = build_records(openmed)
    bt_records = build_records(btsc)
    if btsc.group_of is None:
        bt_records = assign_blocks(bt_records, btsc_block_size)

    om_train, om_val, om_test = _partition_groups(om_records, val_frac, test_frac, rng)
    bt_train, bt_val, bt_test = _partition_groups(bt_records, val_frac, test_frac, rng)

    manifest = SplitManifest(
        train=om_train + bt_train,
        val=om_val + bt_val,
        test=om_test + bt_test,
    )

    # ── verification ─────────────────────────────────────────────────────
    leaked = verify_no_group_leak(manifest)

    counts = {
        "train": _source_counts(manifest.train),
        "val": _source_counts(manifest.val),
        "test": _source_counts(manifest.test),
    }

    # How exposed is the block split? Every block boundary is a place a scan
    # could straddle two splits. Count the slices sitting at such a boundary.
    boundary_exposure = _block_boundary_exposure(manifest)

    report = {
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "btsc_block_size": btsc_block_size,
        "counts": counts,
        "total": {k: sum(v.values()) for k, v in counts.items()},
        "group_leaks": leaked,
        "n_group_leaks": len(leaked),
        "group_kind": {
            "openmed": {"kind": openmed.group_kind,
                        "is_subject_verified": openmed.group_is_subject_verified},
            "btsc": {"kind": "block",
                     "is_subject_verified": False,
                     "block_size": btsc_block_size},
        },
        "btsc_boundary_exposure": boundary_exposure,
        "caveat": (
            "OpenMed groups are lesion keys derived from filenames; BTSC groups are "
            "contiguous index blocks. Neither is a verified patient identifier. "
            "No group appears in more than one split, which eliminates the leakage "
            "that grouping *can* eliminate. Residual risk is limited to BTSC block "
            "boundaries, quantified in btsc_boundary_exposure."
        ),
    }
    manifest.report = report
    return manifest


def _source_counts(records: list[Record]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        counts[rec.source] += 1
    return dict(counts)


def verify_no_group_leak(manifest: SplitManifest) -> list[dict[str, str]]:
    """Return every group that appears in more than one split.

    A group legitimately appears many times *inside* one split — every slice of
    a lesion shares its key, so train alone holds hundreds of rows with the same
    group. A leak is strictly the same group turning up in two different splits,
    which is the only thing this function reports. An empty list is the
    pass condition.
    """
    seen: dict[tuple[str, str], str] = {}
    leaked: list[dict[str, str]] = []
    for split_name, part in (
        ("train", manifest.train), ("val", manifest.val), ("test", manifest.test)
    ):
        for rec in part:
            key = (rec.source, rec.group)
            previous = seen.get(key)
            if previous is None:
                seen[key] = split_name
            elif previous != split_name:
                leaked.append(
                    {"source": rec.source, "group": rec.group,
                     "first": previous, "second": split_name}
                )
    return leaked


def _block_boundary_exposure(manifest: SplitManifest) -> dict[str, Any]:
    """Quantify residual adjacency risk at BTSC block boundaries."""
    btsc = [r for r in manifest.all_records() if r.group.startswith("block_")]
    if not btsc:
        return {"applicable": False}

    split_of = {}
    for name, part in (("train", manifest.train), ("val", manifest.val), ("test", manifest.test)):
        for rec in part:
            split_of[rec.stem] = name

    def as_int(stem: str) -> Optional[int]:
        try:
            return int(stem)
        except ValueError:
            return None

    numeric = sorted((as_int(r.stem), r.stem) for r in btsc if as_int(r.stem) is not None)
    adjacent_cross_split = 0
    for i in range(len(numeric) - 1):
        a, b = numeric[i][1], numeric[i + 1][1]
        if split_of.get(a) != split_of.get(b):
            adjacent_cross_split += 1

    exposed = adjacent_cross_split * 2
    total = len(btsc)
    return {
        "applicable": True,
        "btsc_total_slices": total,
        "adjacent_pairs_crossing_a_split": adjacent_cross_split,
        "slices_touching_a_cross_split_pair": exposed,
        "exposed_fraction": round(exposed / total, 5) if total else None,
        "interpretation": (
            "A slice counted here has a numerically adjacent neighbour in a different "
            "split. Those neighbours may belong to the same scan. This is the residual "
            "risk that cannot be removed without a patient identifier, and it is the "
            "number to quote in the limitations section — not zero."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def _augment(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, cfg: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Joint geometric augmentation. Mirrors are exact; rotation uses reflect-free
    black padding so the result still looks like a cropped acquisition."""
    if rng.random() < cfg.aug_hflip_p:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])

    if rng.random() < cfg.aug_rotate_p:
        angle = float(rng.uniform(-cfg.aug_rotate_deg, cfg.aug_rotate_deg))
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        rotated_image = cv2.warpAffine(
            image, matrix, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        rotated_mask = cv2.warpAffine(
            mask, matrix, (w, h), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        # Reject a rotation that clipped the lesion away.
        before = float((mask > 0).sum())
        after = float((rotated_mask > 0).sum())
        if before == 0 or after / before >= 0.85:
            image, mask = rotated_image, rotated_mask

    return image, mask


def _mask_to_target(mask: np.ndarray, resized_hw: tuple[int, int], size: int) -> np.ndarray:
    """Nearest-neighbour resize of the mask onto the same geometry as the image."""
    new_h, new_w = resized_hw
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    padded = np.zeros((size, size), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    return padded


def mask_to_box(
    mask_target: np.ndarray, margin: int, jitter: int = 0, rng: Optional[np.random.Generator] = None
) -> Optional[np.ndarray]:
    """Ground-truth mask -> xyxy box in target (256) coordinates.

    A positive margin widens the box because the serving pipeline's prompt comes
    from a Grad-CAM heatmap, which is looser than a tight mask bound. Jitter is
    applied on top during training so the model tolerates an imperfect prompt.
    """
    coords = np.argwhere(mask_target > 0)
    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    size = mask_target.shape[0]

    x1 = float(max(0, x_min - margin))
    y1 = float(max(0, y_min - margin))
    x2 = float(min(size - 1, x_max + margin))
    y2 = float(min(size - 1, y_max + margin))

    if jitter > 0 and rng is not None:
        def shift(lo: float, hi: float) -> tuple[float, float]:
            d_lo = float(rng.uniform(-jitter, jitter))
            d_hi = float(rng.uniform(-jitter, jitter))
            return max(0.0, lo + d_lo), min(float(size - 1), hi + d_hi)

        x1, x2 = shift(x1, x2)
        y1, y2 = shift(y1, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:  # a degenerate box teaches nothing
            x1, y1 = float(max(0, x_min - margin)), float(max(0, y_min - margin))
            x2, y2 = float(min(size - 1, x_max + margin)), float(min(size - 1, y_max + margin))

    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TumourSegDataset(Dataset):
    """One image, one binary mask, one box prompt — no class label anywhere.

    Preprocessing is delegated to the serving module's ``preprocess_image`` so
    the tensor the model trains on is produced by the same code that produces
    the tensor it will be served. Any divergence here silently degrades
    inference, so it is deliberately not reimplemented.
    """

    def __init__(
        self,
        records: list[Record],
        cfg: Any,
        *,
        train: bool = False,
        jitter: int = 0,
        seed: int = 0,
    ) -> None:
        self.records = records
        self.cfg = cfg
        self.train = train
        self.jitter = jitter
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Reseed augmentation per epoch so the same slice varies across epochs."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]

        image = cv2.imread(rec.image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unreadable image: {rec.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(rec.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Unreadable mask: {rec.mask_path}")
        mask = (mask > 0).astype(np.uint8)

        if self.train:
            rng = np.random.default_rng((self.seed + index) * 1000003 + self.epoch)
            image, mask = _augment(image, mask, rng, self.cfg)

        # Copied *after* augmentation so it always corresponds to the image the
        # model actually sees. Evaluation never augments, so this is the raw
        # annotation there — which is what postprocess_mask expects.
        native_mask = mask.copy()

        # Serving-identical preprocessing.
        prep = preprocess_image(image, target_size=MEDSAM_INPUT_SIZE)
        mask_target = _mask_to_target(mask, prep.resized_hw, MEDSAM_INPUT_SIZE)

        box = mask_to_box(
            mask_target,
            margin=self.cfg.box_margin_px,
            jitter=self.jitter,
            rng=np.random.default_rng((self.seed + index) * 7919 + self.epoch) if self.jitter else None,
        )
        if box is None:
            # Should be impossible after the audit (no empty masks), but a
            # zero-area prompt would produce a meaningless gradient. Fall back
            # to a centred box rather than crash a 3-hour run.
            size = MEDSAM_INPUT_SIZE
            box = np.array([size * 0.25, size * 0.25, size * 0.75, size * 0.75], dtype=np.float32)

        valid_h, valid_w = prep.resized_hw
        return {
            "image": prep.tensor.squeeze(0),                      # (3, 256, 256)
            "mask_target": torch.from_numpy(mask_target).float(),  # (256, 256) in {0,1}
            "box": torch.from_numpy(box).reshape(1, 4),            # (1, 4)
            "native_mask": torch.from_numpy(native_mask).float(),  # (H, W) in {0,1}
            "valid_hw": torch.tensor([valid_h, valid_w], dtype=torch.int64),
            "source": rec.source,
            "stem": rec.stem,
            "group": rec.group,
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors, keep variable-size native masks as a list."""
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "mask_target": torch.stack([b["mask_target"] for b in batch], dim=0).unsqueeze(1),
        "box": torch.stack([b["box"] for b in batch], dim=0),
        "native_mask": [b["native_mask"] for b in batch],
        "valid_hw": torch.stack([b["valid_hw"] for b in batch], dim=0),
        "source": [b["source"] for b in batch],
        "stem": [b["stem"] for b in batch],
        "group": [b["group"] for b in batch],
    }
