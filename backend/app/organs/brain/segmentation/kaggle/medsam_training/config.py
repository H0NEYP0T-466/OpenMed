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


_DATASET_MARKERS = ("images", "png_dataset", "labels")
# Directories that *contain* the data. Pruned during the walk so we never
# descend into 10,000 image files looking for a nested dataset.
_CONTENT_DIRS = ("images", "masks", "labels", "png_dataset")


def _looks_like_dataset(path: str) -> bool:
    return os.path.isdir(path) and any(
        os.path.isdir(os.path.join(path, marker)) for marker in _DATASET_MARKERS
    )


def _dataset_dirs(root: str, max_depth: int = 4) -> list[str]:
    """Every dataset-shaped directory at or below ``root``, breadth-first.

    Kaggle nests mounts inconsistently — sometimes ``/kaggle/input/<slug>``,
    sometimes ``/kaggle/input/datasets/<owner>/<slug>``. Rather than hard-code
    either shape, walk a bounded depth and collect everything that looks like a
    dataset. Depth is capped so a stray symlink cannot turn this into a crawl.
    """
    found: list[str] = []
    if not os.path.isdir(root):
        return found

    base_depth = os.path.abspath(root).rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, _filenames in os.walk(root):
        depth = dirpath.rstrip(os.sep).count(os.sep) - base_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(d for d in dirnames if d not in _CONTENT_DIRS and not d.startswith("."))
        if _looks_like_dataset(dirpath):
            found.append(dirpath)
    return found


def _images_dir_of(path: str) -> Optional[str]:
    for marker in _DATASET_MARKERS:
        if os.path.isdir(os.path.join(path, marker)):
            return marker
    return None


def _classify_dataset(path: str, sample: int = 400) -> Optional[str]:
    """Guess which source a directory holds from its filenames.

    The two datasets are trivially separable by naming convention: BTSC's PNG
    export is ``1.png … 3064.png`` (all numeric stems), OpenMed's files are
    descriptive (``T1C+ - Cystic glioblastoma occipital , ventricle 005``).
    That makes a name-independent fallback possible when the mounted slug
    matches nothing in the candidate lists.
    """
    images_dir = _images_dir_of(path)
    if images_dir is None:
        return None
    try:
        names = [
            n for n in os.listdir(os.path.join(path, images_dir))[:sample]
            if not n.startswith(".")
        ]
    except OSError:
        return None

    stems = [os.path.splitext(n)[0] for n in names]
    if not stems:
        return None

    numeric = sum(1 for s in stems if s.isdigit())
    ratio = numeric / len(stems)
    if ratio >= 0.9:
        return "btsc"
    if ratio <= 0.1:
        return "openmed"
    return None


def _locate(kind: str, candidates: tuple[str, ...], root: Optional[str] = None) -> Optional[str]:
    """Find one dataset, trying three strategies in order of confidence."""
    root = root or KAGGLE_INPUT
    wanted = {c.lower() for c in candidates}

    # 1. A direct child whose name matches — the common /kaggle/input/<slug> case.
    for name in candidates:
        path = os.path.join(root, name)
        if _looks_like_dataset(path):
            return path

    dirs = _dataset_dirs(root)

    # 2. Any depth, matched by directory name. Handles
    #    /kaggle/input/datasets/<owner>/<slug> without needing the owner.
    for path in dirs:
        if os.path.basename(path).lower() in wanted:
            return path

    # 3. Last resort: infer from the filenames themselves.
    for path in dirs:
        if _classify_dataset(path) == kind:
            return path

    return None


def resolve_paths(cfg: TrainConfig) -> TrainConfig:
    """Fill in dataset roots from the Kaggle mount points when not given.

    Raises a single, actionable error listing what was actually found — a
    Kaggle notebook failing at cell 1 with a clear message beats failing at
    epoch 40.
    """
    if cfg.openmed_root is None:
        cfg.openmed_root = _locate("openmed", OPENMED_CANDIDATES)
    if cfg.btsc_root is None:
        cfg.btsc_root = _locate("btsc", BTSC_CANDIDATES)

    if cfg.openmed_root and cfg.btsc_root and cfg.openmed_root == cfg.btsc_root:
        raise FileNotFoundError(
            f"Both sources resolved to the same directory: {cfg.openmed_root}. "
            "Pass --openmed-root and --btsc-root explicitly."
        )

    missing = []
    if not cfg.openmed_root:
        missing.append(f"OpenMed dataset (names tried: {OPENMED_CANDIDATES})")
    if not cfg.btsc_root:
        missing.append(f"BTSC dataset (names tried: {BTSC_CANDIDATES})")
    if missing:
        discovered = _dataset_dirs(KAGGLE_INPUT)
        listing = (
            "\n".join(
                f"    {p}  -> looks like: {_classify_dataset(p) or 'unrecognised'}"
                for p in discovered
            )
            or "    (none found)"
        )
        raise FileNotFoundError(
            "Could not locate dataset root(s): " + "; ".join(missing) + ".\n"
            f"  Searched under {KAGGLE_INPUT} (depth 4). Dataset-shaped directories found:\n"
            f"{listing}\n"
            "  Fix: attach the datasets in the notebook's Data panel, or pass\n"
            "  --openmed-root / --btsc-root with the paths printed above."
        )
    return cfg
