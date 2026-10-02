"""Central configuration for LiteMedSAM fine-tuning.

Everything tunable lives here so ``trainKaggle.py`` stays a thin orchestrator
and every stage module takes an explicit, serialisable config dict.

Defaults are chosen for **Kaggle 2 x Tesla T4 (15 GB each)** with the two
mounted datasets described in the README:

    /kaggle/input/fypseg                    (OpenMed segmentation, 10,056 pairs)
    /kaggle/input/brain-tumor-segmentation  (figshare/BTSC, 3,064 pairs)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# ── Kaggle mount points ──────────────────────────────────────────────────────
KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORKING = "/kaggle/working"

# Candidate directory names for each source, tried in order. The first one that
# exists and contains an images/ directory wins. Kept as a list because Kaggle
# slugifies dataset titles differently between uploads.
OPENMED_CANDIDATES = (
    "fypseg",
    "openmed",
    "openmed-seg",
    "openmed-seg-clean-strict",
    "openmed_seg_clean_strict",
)
BTSC_CANDIDATES = (
    "brain-tumor-segmentation",
    "brain_tumor_segmentation",
    "btsc",
    "btsc-unet-seg",
    "btsc-unet-dataset",
)

# Official LiteMedSAM release (bowang-lab). Same file ID the backend uses in
# prepare_weights.py — kept in sync deliberately so training and serving start
# from identical weights.
GDRIVE_FILE_ID = "18Zed-TUTsmr2zc5CHUWd5Tu13nb6vq6z"
WEIGHT_FILENAME = "lite_medsam.pth"
EXPECTED_MIN_BYTES = 30_000_000

# Model input geometry. Must equal preprocessor.MEDSAM_INPUT_SIZE.
IMAGE_SIZE = 256


@dataclass
class TrainConfig:
    """Full run configuration. Serialised to config.json for reproducibility."""

    # ── paths ────────────────────────────────────────────────────────────
    openmed_root: Optional[str] = None
    btsc_root: Optional[str] = None
    out_dir: str = os.path.join(KAGGLE_WORKING, "openmed_seg_run")
    weights_dir: str = os.path.join(KAGGLE_WORKING, "weights")
    pretrained: Optional[str] = None          # resolved during bootstrap

    # ── data ─────────────────────────────────────────────────────────────
    seed: int = 42
    # Split fractions. Applied to *groups*, never to individual slices.
    val_frac: float = 0.10
    test_frac: float = 0.10
    # BTSC has no patient identifier in the PNG export, so its ordering is
    # split into contiguous blocks. Block size in slices. Smaller blocks mean
    # finer-grained splits but more block boundaries (each boundary is a
    # possible scan straddle). 32 keeps boundary risk near 2 % of slices.
    btsc_block_size: int = 32
    # OpenMed group key: sequence-stripped, slice-number-stripped stem. This is
    # a *lesion* key derived from the dataset's own naming, not a proven
    # patient ID — recorded as such in the audit output.
    openmed_group_is_subject_verified: bool = False

    # ── optimisation ─────────────────────────────────────────────────────
    epochs: int = 100
    batch_size: int = 8            # per GPU; 2 GPUs -> effective 16
    grad_accum: int = 1
    workers: int = 2
    lr: float = 1e-4               # decoder / prompt LR
    encoder_lr: float = 1e-5       # TinyViT LR (10x lower)
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    min_lr_ratio: float = 0.05
    grad_clip: float = 1.0
    amp: bool = True
    grad_checkpoint: bool = False  # TinyViT gradient checkpointing
    freeze_prompt_encoder: bool = True
    freeze_bn: bool = True

    # ── loss ─────────────────────────────────────────────────────────────
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    iou_head_weight: float = 0.05
    # Restrict loss to the unpadded region. Padding is a constant zero block
    # in the bottom-right; supervising it teaches the model a positional prior
    # that the serving pipeline reproduces anyway, but wastes capacity.
    loss_on_valid_region_only: bool = True

    # ── prompts ──────────────────────────────────────────────────────────
    box_margin_px: int = 5          # GT mask -> box, in 256-space
    box_jitter_px: int = 10         # random jitter, train only
    eval_jitter_px: int = 6         # deterministic perturbed prompt for stress test

    # ── augmentation ─────────────────────────────────────────────────────
    aug_hflip_p: float = 0.5
    aug_rotate_p: float = 0.25
    aug_rotate_deg: float = 10.0

    # ── early stopping ───────────────────────────────────────────────────
    patience: int = 10
    min_delta: float = 1e-4
    monitor: str = "val_macro_dice"

    # ── hygiene thresholds ───────────────────────────────────────────────
    align_min_inside: float = 0.90   # mask foreground on anatomy
    phash_hamming_max: int = 3       # calibrated on these datasets
    phash_sample: int = 4000         # 0 = all; sampling keeps the audit fast

    # ── runtime ──────────────────────────────────────────────────────────
    master_port: str = "29500"
    gpus: int = 2
    resume: Optional[str] = None
    max_steps_per_epoch: int = 0     # 0 = unlimited; >0 for debugging only
    smoke: bool = False              # tiny end-to-end run, for CI only

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        with open(path) as fh:
            return cls(**json.load(fh))

    def fingerprint(self) -> str:
        """Stable hash of the fields that must match for a resume to be valid."""
        import hashlib

        keys = (
            "seed", "val_frac", "test_frac", "btsc_block_size", "epochs",
            "batch_size", "grad_accum", "lr", "encoder_lr", "weight_decay",
            "warmup_epochs", "bce_weight", "dice_weight", "iou_head_weight",
            "box_margin_px", "box_jitter_px", "openmed_root", "btsc_root",
        )
        payload = json.dumps({k: getattr(self, k) for k in keys}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _first_existing(
    candidates: tuple[str, ...], markers: tuple[str, ...] = ("images", "png_dataset")
) -> Optional[str]:
    """Return the first ``/kaggle/input/<name>`` that exists and holds a dataset.

    Falls back to a recursive search when the mounted slug matches none of the
    candidates, because Kaggle nests datasets as ``/kaggle/input/<slug>`` and the
    slug is set by the uploader. Multiple markers are checked because the BTSC
    mirror may keep the upstream ``png_dataset/`` layout instead of
    ``images/`` + ``masks/``.
    """
    def looks_like_dataset(path: str) -> bool:
        return any(os.path.isdir(os.path.join(path, marker)) for marker in markers)

    for name in candidates:
        path = os.path.join(KAGGLE_INPUT, name)
        if looks_like_dataset(path):
            return path

    if os.path.isdir(KAGGLE_INPUT):
        for entry in sorted(os.listdir(KAGGLE_INPUT)):
            path = os.path.join(KAGGLE_INPUT, entry)
            if looks_like_dataset(path):
                return path
    return None


def resolve_paths(cfg: TrainConfig) -> TrainConfig:
    """Fill in dataset roots from the Kaggle mount points when not given.

    Raises a single, actionable error listing what was searched — a Kaggle
    notebook failing at cell 1 with a clear message beats failing at epoch 40.
    """
    if cfg.openmed_root is None:
        cfg.openmed_root = _first_existing(OPENMED_CANDIDATES)
    if cfg.btsc_root is None:
        cfg.btsc_root = _first_existing(BTSC_CANDIDATES)

    missing = []
    if not cfg.openmed_root:
        missing.append(
            f"OpenMed dataset (expected one of {OPENMED_CANDIDATES} under {KAGGLE_INPUT})"
        )
    if not cfg.btsc_root:
        missing.append(
            f"BTSC dataset (expected one of {BTSC_CANDIDATES} under {KAGGLE_INPUT})"
        )
    if missing:
        found = sorted(os.listdir(KAGGLE_INPUT)) if os.path.isdir(KAGGLE_INPUT) else []
        raise FileNotFoundError(
            "Could not locate dataset root(s): " + "; ".join(missing) + ". "
            f"Contents of {KAGGLE_INPUT}: {found}. "
            "Pass --openmed-root / --btsc-root explicitly if the slug differs."
        )
    return cfg
