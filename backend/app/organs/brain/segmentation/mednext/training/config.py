"""Central configuration for MedNeXt training.

Everything tunable lives here so ``train_mednext.py`` stays a thin
orchestrator and every stage takes an explicit, serialisable config. Defaults
target Kaggle 2 x Tesla T4 (15 GB each) with the two mounted datasets:

    /kaggle/input/.../fypseg                    OpenMed segmentation
    /kaggle/input/.../brain-tumor-segmentation  figshare / BTSC
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional

KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORKING = "/kaggle/working"
KAGGLE_TEMP = "/kaggle/temp"

CHECKPOINT_NAME = "mednext_brain_seg.pth"

OPENMED_CANDIDATES = (
    "fypseg",
    "openmed",
    "openmed-seg",
    "openmed-seg-clean-strict",
    "openmed_seg_clean_strict",
    "archive",
)
BTSC_CANDIDATES = (
    "brain-tumor-segmentation",
    "brain_tumor_segmentation",
    "btsc",
    "btsc-unet-seg",
    "btsc-unet-dataset",
    "unet_tumor_dataset",
)

_DATASET_MARKERS = ("images", "png_dataset", "labels")
_CONTENT_DIRS = ("images", "masks", "labels", "png_dataset")


@dataclass
class TrainConfig:
    """Full run configuration. Serialised to ``config.json`` for reproducibility."""

    # ── paths ────────────────────────────────────────────────────────────
    openmed_root: Optional[str] = None
    btsc_root: Optional[str] = None
    out_dir: str = os.path.join(KAGGLE_WORKING, "mednext_run")
    cache_dir: Optional[str] = None
    zip_dir: str = KAGGLE_WORKING

    # ── data and split ───────────────────────────────────────────────────
    seed: int = 42
    image_size: int = 256
    val_frac: float = 0.10
    test_frac: float = 0.10
    # BTSC ships without a patient id. Its index order is cut into blocks of
    # about this many slices, with each cut snapped to the weakest adjacent-slice
    # similarity within +-btsc_snap_window (a likely scan boundary).
    btsc_mode: str = "block"            # block | train_only | external_test
    btsc_block_size: int = 32
    btsc_snap_window: int = 8
    btsc_purge_radius: int = 2
    btsc_purge_corr: float = 0.90
    group_merge_cap_frac: float = 0.10
    split_search_tries: int = 400

    limit_per_source: int = 0           # debugging only: first N stems of each source

    # ── hygiene thresholds ───────────────────────────────────────────────
    align_min_inside: float = 0.90
    phash_hamming_max: int = 3
    phash_dup_hamming: int = 2
    dup_pixel_mae: float = 3.0
    min_image_std: float = 8.0
    tiny_mask_frac: float = 0.0002

    # ── model ────────────────────────────────────────────────────────────
    variant: str = "B"
    kernel_size: int = 3
    grn: bool = False
    drop_path: float = 0.05
    deep_supervision: bool = True
    grad_checkpoint: bool = False
    upkern_from: Optional[str] = None

    # ── optimisation ─────────────────────────────────────────────────────
    epochs: int = 150
    batch_size: int = 16               # per GPU
    grad_accum: int = 1
    workers: int = 3
    lr: float = 5e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 3
    min_lr_ratio: float = 0.01
    grad_clip: float = 1.0
    amp: bool = True
    amp_init_scale: Optional[int] = 1024
    ema_decay: float = 0.999
    ema_start_step: int = 200

    # ── loss ─────────────────────────────────────────────────────────────
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    ds_decay: float = 0.5              # deep-supervision weight of head i is decay**i

    # ── augmentation (probabilities per sample) ──────────────────────────
    aug_enabled: bool = True
    aug_hflip_p: float = 0.5
    aug_affine_p: float = 0.7
    aug_rotate_deg: float = 15.0
    aug_scale_range: tuple[float, float] = (0.85, 1.15)
    aug_translate_frac: float = 0.06
    aug_shear_deg: float = 4.0
    aug_elastic_p: float = 0.2
    aug_gamma_p: float = 0.3
    aug_gamma_range: tuple[float, float] = (0.7, 1.5)
    aug_bias_field_p: float = 0.3
    aug_noise_p: float = 0.25
    aug_blur_p: float = 0.2
    aug_jpeg_p: float = 0.25
    aug_lowres_p: float = 0.1

    # ── early stopping ───────────────────────────────────────────────────
    patience: int = 10
    min_delta: float = 1e-4

    # ── post-processing search (on validation) ───────────────────────────
    pp_thresholds: tuple[float, ...] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
    pp_min_area_fracs: tuple[float, ...] = (0.0, 0.0002, 0.0005, 0.001, 0.002)
    tta_hflip: bool = True

    # ── runtime ──────────────────────────────────────────────────────────
    gpus: int = 2
    master_port: str = "29517"
    resume: Optional[str] = None
    time_budget_hours: float = 10.5
    max_steps_per_epoch: int = 0
    smoke: bool = False
    export_zip: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainConfig:
        names = {f.name for f in fields(cls)}
        data = {k: v for k, v in payload.items() if k in names}
        for key in ("aug_scale_range", "aug_gamma_range", "pp_thresholds", "pp_min_area_fracs"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        return cls(**data)

    def fingerprint(self) -> str:
        """Hash of the fields that must match for a resume to be meaningful."""
        keys = (
            "seed", "image_size", "val_frac", "test_frac", "btsc_mode", "btsc_block_size",
            "variant", "kernel_size", "grn", "drop_path", "deep_supervision",
            "epochs", "batch_size", "grad_accum", "lr", "weight_decay", "warmup_epochs",
            "bce_weight", "dice_weight", "ema_decay", "gpus",
        )
        payload = json.dumps({k: getattr(self, k) for k in keys}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def resolved_cache_dir(self) -> str:
        if self.cache_dir:
            return self.cache_dir
        if os.path.isdir(KAGGLE_TEMP) and os.access(KAGGLE_TEMP, os.W_OK):
            return os.path.join(KAGGLE_TEMP, "mednext_cache")
        import tempfile

        return os.path.join(tempfile.gettempdir(), "mednext_cache")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery
# ─────────────────────────────────────────────────────────────────────────────

def _looks_like_dataset(path: str) -> bool:
    return os.path.isdir(path) and any(
        os.path.isdir(os.path.join(path, marker)) for marker in _DATASET_MARKERS
    )


def dataset_dirs(root: str, max_depth: int = 4) -> list[str]:
    """Every dataset-shaped directory at or below ``root``, breadth-first.

    Kaggle nests mounts inconsistently (``/kaggle/input/<slug>`` or
    ``/kaggle/input/datasets/<owner>/<slug>``), so a bounded walk is used and
    content directories are pruned to avoid crawling tens of thousands of files.
    """
    found: list[str] = []
    if not os.path.isdir(root):
        return found
    base_depth = os.path.abspath(root).rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, _ in os.walk(root):
        depth = dirpath.rstrip(os.sep).count(os.sep) - base_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(
            d for d in dirnames if d not in _CONTENT_DIRS and not d.startswith(".")
        )
        if _looks_like_dataset(dirpath):
            found.append(dirpath)
    return found


def _images_dir_of(path: str) -> Optional[str]:
    for marker in _DATASET_MARKERS:
        if os.path.isdir(os.path.join(path, marker)):
            return marker
    return None


def classify_dataset(path: str, sample: int = 400) -> Optional[str]:
    """Guess ``btsc`` or ``openmed`` from filenames: BTSC stems are all numeric."""
    images_dir = _images_dir_of(path)
    if images_dir is None:
        return None
    try:
        names = [
            n for n in sorted(os.listdir(os.path.join(path, images_dir)))[:sample]
            if not n.startswith(".")
        ]
    except OSError:
        return None
    stems = [os.path.splitext(n)[0] for n in names]
    if not stems:
        return None
    ratio = sum(1 for s in stems if s.isdigit()) / len(stems)
    if ratio >= 0.9:
        return "btsc"
    if ratio <= 0.1:
        return "openmed"
    return None


def _locate(kind: str, candidates: tuple[str, ...], root: str) -> Optional[str]:
    wanted = {c.lower() for c in candidates}

    def acceptable(path: str) -> bool:
        guess = classify_dataset(path)
        return guess is None or guess == kind

    for name in candidates:
        path = os.path.join(root, name)
        if _looks_like_dataset(path) and acceptable(path):
            return path

    found = dataset_dirs(root)
    for path in found:
        if os.path.basename(path).lower() in wanted and acceptable(path):
            return path

    base = os.path.abspath(root).rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, _ in os.walk(root):
        if dirpath.rstrip(os.sep).count(os.sep) - base > 4:
            dirnames[:] = []
            continue
        if os.path.basename(dirpath).lower() not in wanted:
            continue
        for child in sorted(dirnames):
            child_path = os.path.join(dirpath, child)
            if _looks_like_dataset(child_path) and acceptable(child_path):
                return child_path

    for path in found:
        if classify_dataset(path) == kind:
            return path
    return None


def resolve_paths(cfg: TrainConfig, input_root: str = KAGGLE_INPUT) -> TrainConfig:
    """Fill in dataset roots from the Kaggle mounts, with one actionable error."""
    if cfg.openmed_root is None:
        cfg.openmed_root = _locate("openmed", OPENMED_CANDIDATES, input_root)
    if cfg.btsc_root is None:
        cfg.btsc_root = _locate("btsc", BTSC_CANDIDATES, input_root)

    if cfg.openmed_root and cfg.btsc_root and (
        os.path.abspath(cfg.openmed_root) == os.path.abspath(cfg.btsc_root)
    ):
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
        listing = "\n".join(
            f"    {p}  -> looks like: {classify_dataset(p) or 'unrecognised'}"
            for p in dataset_dirs(input_root)
        ) or "    (none found)"
        raise FileNotFoundError(
            "Could not locate dataset root(s): " + "; ".join(missing) + ".\n"
            f"  Searched under {input_root} (depth 4). Dataset-shaped directories found:\n"
            f"{listing}\n"
            "  Fix: attach the datasets in the notebook's Data panel, or pass\n"
            "  --openmed-root / --btsc-root with the paths printed above."
        )
    return cfg
