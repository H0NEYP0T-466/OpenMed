"""Central configuration and dataset discovery for lungs segmentation training.

Everything tunable lives here so ``train_lungs_seg.py`` stays a thin orchestrator
and every stage takes an explicit, serialisable config. Defaults target Kaggle
2 x Tesla T4 (15 GB each).

Dataset discovery never hardcodes a path. On Kaggle the data is a mounted dataset
whose nesting varies, so a bounded breadth-first walk looks for the two COVID-QU-Ex
trees (each has a doubled directory level and ``Train/Val/Test`` below it):

    Lung Segmentation Data/Lung Segmentation Data/{Train,Val,Test}/<class>/{images,lung masks}
    Infection Segmentation Data/Infection Segmentation Data/{...}/<class>/{images,lung masks,infection masks}
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from typing import Any

KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORKING = "/kaggle/working"
KAGGLE_TEMP = "/kaggle/temp"

STAGES = ("lung", "twohead")
SPLITS = ("train", "val", "test")
SHIPPED_SPLIT_DIRS = {"train": "Train", "val": "Val", "test": "Test"}

IMAGES_DIR = "images"
LUNG_MASK_DIR = "lung masks"
INFECTION_MASK_DIR = "infection masks"
_PRUNE = {IMAGES_DIR, LUNG_MASK_DIR, INFECTION_MASK_DIR}


@dataclass
class TrainConfig:
    """Full run configuration. Serialised to ``config.json`` for reproducibility."""

    # -- paths ----------------------------------------------------------------------
    data_root: str | None = None
    out_dir: str = ""
    cache_dir: str | None = None
    zip_dir: str = KAGGLE_WORKING
    split_from: str | None = None       # reuse another run's split_manifest.csv
    init_from: str | None = None        # serving/best checkpoint for a WEIGHTS-ONLY start
    pretrained: str = "nvidia/mit-b0"   # HF hub id or local directory; "" = random init
    allow_random_init: bool = False

    # -- stage ----------------------------------------------------------------------
    stage: str = "lung"                 # lung | twohead

    # -- data, hygiene and split ----------------------------------------------------
    seed: int = 42
    image_size: int = 256
    split_fracs: tuple[float, float, float] = (0.64, 0.16, 0.20)
    limit_per_class: int = 0            # debugging only: first N pairs per class and split
    lung_area_floor_px: int = 2725      # smallest real lung region seen in the dataset
    full_frame_fraction: float = 0.98
    blank_std: float = 3.0
    low_contrast_std: float = 12.0
    saturated_fraction: float = 0.5
    align_inside_min: float = 0.90      # share of lung mask on non-padding pixels
    border_band_px: int = 4
    border_fraction_warn: float = 0.05
    phash_hamming_max: int = 4
    phash_pixel_mae: float = 6.0
    phash_min_corr: float = 0.98
    phash_bucket_cap: int = 400         # skip degenerate buckets (blank images collide)
    group_by_subject: bool = True       # BIDS-style names carry a subject id: one patient, one split
    continue_on_fail: bool = False      # drop FAIL violators instead of aborting

    # -- model ----------------------------------------------------------------------
    encoder: str = "mit-b0"             # mit-b0 | debug-tiny (smoke tests only)
    drop_path: float = 0.1
    decoder_dropout: float = 0.1
    head_dropout: float = 0.1

    # -- optimisation ---------------------------------------------------------------
    epochs: int = 100
    batch_size: int = 32                # per GPU
    grad_accum: int = 1
    workers: int = 3
    lr: float = 5e-4                    # decoder and heads
    encoder_lr_mult: float = 0.2        # pretrained encoder trains slower than the new decoder
    weight_decay: float = 0.05          # decoupled (AdamW); norm and bias exempt
    warmup_epochs: int = 3
    min_lr_ratio: float = 0.01
    grad_clip: float = 1.0
    amp: bool = True                    # fp16: T4 is sm_75, no bf16 tensor cores
    amp_init_scale: int = 1024
    ema_decay: float = 0.999
    ema_start_step: int = 200
    patience: int = 15
    min_delta: float = 1e-4

    # -- loss -----------------------------------------------------------------------
    lung_weight: float = 1.0
    opacity_weight: float = 1.0
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    dice_eps: float = 1.0

    # -- sampler --------------------------------------------------------------------
    min_supervised_per_batch: int = 2
    normal_neg_weight: float = 1.0      # >1 oversamples supervised Normal (true negatives)

    # -- model selection ------------------------------------------------------------
    # opacity score = positive Dice - fpr_weight * Normal false-positive rate, so a
    # head that paints opacity everywhere cannot win on Dice alone.
    fpr_weight: float = 0.25
    selection_lung_share: float = 0.5   # twohead score = share*lung + (1-share)*opacity

    # -- augmentation (probabilities are per sample; strength scales every magnitude) --
    aug_enabled: bool = True
    aug_strength: float = 1.0
    aug_hflip_p: float = 0.5
    aug_affine_p: float = 0.7
    aug_rotate_deg: float = 10.0
    aug_scale_range: tuple[float, float] = (0.9, 1.1)
    aug_translate_frac: float = 0.05
    aug_contrast_p: float = 0.6
    aug_brightness: float = 0.15
    aug_contrast: float = 0.2
    aug_gamma_p: float = 0.3
    aug_gamma_range: tuple[float, float] = (0.8, 1.25)
    aug_jpeg_p: float = 0.25
    aug_jpeg_quality: tuple[int, int] = (40, 90)
    aug_lowres_p: float = 0.15
    aug_lowres_scale: tuple[float, float] = (0.5, 0.85)
    crop_jitter: float = 0.03           # training-time jitter on the lung-bbox crop
    crop_margin_jitter: float = 0.03

    # -- post-processing search (validation) ----------------------------------------
    pp_thresholds: tuple[float, ...] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
    pp_lung_min_area_fracs: tuple[float, ...] = (0.0, 0.0005, 0.001, 0.002, 0.005)
    pp_keep_k: tuple[int, ...] = (1, 2, 0)
    pp_opacity_min_area_fracs: tuple[float, ...] = (0.0, 0.0002, 0.0005, 0.001, 0.002, 0.005)
    pp_tune_max_images: int = 2500      # validation images used for the lung grid
    tta_hflip: bool = True

    # -- regularisation probe (short runs on validation, then pick) -------------------
    probe: bool = True
    probe_epochs: int = 4
    probe_train_frac: float = 0.35

    # -- runtime --------------------------------------------------------------------
    gpus: int = 2
    master_port: str = "29531"
    resume: str = "auto"                # auto | none | <path to last.pt>
    force_resume: bool = False
    time_budget_hours: float = 10.5
    max_steps_per_epoch: int = 0
    smoke: bool = False
    export_zip: bool = True
    allow_retest: bool = False
    is_probe: bool = False              # set on probe sub-runs only

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainConfig:
        names = {f.name for f in fields(cls)}
        data = {k: v for k, v in payload.items() if k in names}
        for key, value in list(data.items()):
            if isinstance(value, list):
                data[key] = tuple(value)
        return cls(**data)

    def fingerprint(self) -> str:
        """Hash of the fields that must match for a resume to be meaningful."""
        keys = (
            "stage", "seed", "image_size", "split_fracs", "group_by_subject", "encoder", "drop_path",
            "decoder_dropout", "head_dropout", "epochs", "batch_size", "grad_accum", "lr",
            "encoder_lr_mult", "weight_decay", "warmup_epochs", "ema_decay", "gpus",
            "lung_weight", "opacity_weight", "bce_weight", "dice_weight", "normal_neg_weight",
            "aug_strength",
        )
        payload = json.dumps({k: getattr(self, k) for k in keys}, sort_keys=True, default=list)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def heads(self) -> tuple[str, ...]:
        return ("lung",) if self.stage == "lung" else ("lung", "opacity")

    def resolved_cache_dir(self) -> str:
        if self.cache_dir:
            return self.cache_dir
        if os.path.isdir(KAGGLE_TEMP) and os.access(KAGGLE_TEMP, os.W_OK):
            return os.path.join(KAGGLE_TEMP, "lungs_seg_cache")
        return os.path.join(tempfile.gettempdir(), "lungs_seg_cache")

    def resolved_out_dir(self) -> str:
        if self.out_dir:
            return self.out_dir
        base = KAGGLE_WORKING if os.path.isdir(KAGGLE_WORKING) else os.getcwd()
        return os.path.join(base, f"lungs_seg_{self.stage}")


# ----------------------------------------------------------------------- discovery


def _child_names(path: str) -> dict[str, str]:
    try:
        return {name.lower(): name for name in os.listdir(path)}
    except OSError:
        return {}


def _is_tree_root(path: str) -> str | None:
    """``"lung"`` / ``"infection"`` if ``path`` holds Train/Val/Test class trees, else ``None``."""
    children = _child_names(path)
    if not all(split in children for split in ("train", "val", "test")):
        return None
    train = os.path.join(path, children["train"])
    classes = [d for d in os.listdir(train) if os.path.isdir(os.path.join(train, d))]
    if not classes:
        return None
    probe = os.path.join(train, classes[0])
    probe_children = _child_names(probe)
    if IMAGES_DIR not in probe_children or LUNG_MASK_DIR not in probe_children:
        return None
    return "infection" if INFECTION_MASK_DIR in probe_children else "lung"


def find_trees(root: str, max_depth: int = 6) -> dict[str, str]:
    """Breadth-first bounded walk for the lung and infection trees under ``root``."""
    found: dict[str, str] = {}
    if not os.path.isdir(root):
        return found
    base = os.path.abspath(root).rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, _ in os.walk(root):
        depth = dirpath.rstrip(os.sep).count(os.sep) - base
        kind = _is_tree_root(dirpath)
        if kind and kind not in found:
            found[kind] = dirpath
            dirnames[:] = []
            continue
        if depth >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(
            d for d in dirnames if d.lower() not in _PRUNE and not d.startswith(".")
        )
        if len(found) == 2:
            break
    return found


def resolve_data_roots(cfg: TrainConfig, search_roots: tuple[str, ...] | None = None) -> dict[str, str]:
    """Locate both trees, with one actionable error that lists what was searched."""
    roots = [cfg.data_root] if cfg.data_root else list(search_roots or (KAGGLE_INPUT,))
    found: dict[str, str] = {}
    for root in roots:
        for kind, path in find_trees(root).items():
            found.setdefault(kind, path)
    missing = [kind for kind in ("lung", "infection") if kind not in found]
    if missing:
        raise FileNotFoundError(
            f"Could not locate the COVID-QU-Ex {' and '.join(missing)} segmentation tree(s) under "
            f"{', '.join(roots)} (bounded walk, depth 6). Expected <root>/Train|Val|Test/<class>/"
            "images + 'lung masks' (+ 'infection masks' for the infection tree). Attach the dataset "
            "in the notebook's Data panel or pass --data-root."
        )
    return found


def class_dirs(tree_root: str, split_dir: str) -> dict[str, str]:
    """Class directory per canonical class name for one split of a tree."""
    from ..labels import normalise_class

    base = os.path.join(tree_root, split_dir)
    out: dict[str, str] = {}
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if os.path.isdir(path):
            out[normalise_class(name)] = path
    return out


def split_dir_names(tree_root: str) -> dict[str, str]:
    children = _child_names(tree_root)
    return {split: children[split] for split in SPLITS}
