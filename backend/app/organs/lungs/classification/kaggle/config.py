"""Configuration, dataset discovery and hyperparameters for the LungV2S trainer.

Everything the training package can tune lives here so ``train_lungs_v2s.py``
stays an orchestrator. The defaults are the contract values, not suggestions:

    backbone      tf_efficientnetv2_s.in21k_ft_in1k (timm)
    task          4-class single-label softmax
    train res     300
    schedule      AdamW 3e-4 / wd 1e-4 / OneCycle pct_start 0.25
    loss          class-weighted CE (inverse frequency, mean 1) + LS 0.1

Dataset layout discovery is deliberately tolerant. The COVID-19 Radiography
Database is mounted in at least three different shapes depending on who
uploaded it, so ``resolve_dataset_root`` walks ``/kaggle/input`` to depth four
and looks for the directory that *directly* contains the four class folders:

    /kaggle/input/<slug>/COVID-19_Radiography_Dataset/{COVID,Lung_Opacity,...}/images/*.png
    /kaggle/input/datasets/<owner>/<slug>/{COVID,Lung_Opacity,...}/*.png

Class folder names are matched case-insensitively and tolerate spaces,
underscores and hyphens (``Lung Opacity`` == ``Lung_Opacity``).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORKING = "/kaggle/working"

MODEL_TAG = "tf_efficientnetv2_s.in21k_ft_in1k"

# Fixed order. Index i of the model's logits is CLASS_NAMES[i], and the sidecar,
# the serving pipeline and the confusion matrices all rely on that.
CLASS_NAMES: tuple[str, ...] = (
    "COVID",
    "Lung_Opacity",
    "Normal",
    "Viral Pneumonia",
)
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_INDEX: dict[str, int] = {name: index for index, name in enumerate(CLASS_NAMES)}

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})

# Minority class that gets a second, independent augmentation draw per sample.
MINORITY_CLASS = "Viral Pneumonia"

# Provenance pools, used by the leakage audit. A pool that appears in more than
# one split is reported; it is expected (the source corpus is shared) but it is
# never allowed to be a *cluster* spanning splits.
SOURCE_POOL_DEFAULTS: dict[str, str] = {
    "COVID": "covid_mixed",
    "Lung_Opacity": "rsna",
    "Normal": "rsna",
    "Viral Pneumonia": "viral",
}

_CLASS_ALIASES: dict[str, tuple[str, ...]] = {
    "COVID": ("covid", "covid19", "covid-19", "covid 19", "covid_19"),
    "Lung_Opacity": (
        "lung_opacity",
        "lung opacity",
        "lungopacity",
        "lung-opacity",
        "opacity",
    ),
    "Normal": ("normal", "normals"),
    "Viral Pneumonia": (
        "viral pneumonia",
        "viral_pneumonia",
        "viralpneumonia",
        "viral-pneumonia",
        "pneumonia viral",
    ),
}

# Directories that never contain class folders and must not be walked into.
_SKIP_DIRS = frozenset({"images", "masks", "mask", "__macosx", ".git", ".ipynb_checkpoints"})


def normalise_name(name: str) -> str:
    """Lowercase and collapse separators so ``Lung_Opacity`` == ``lung opacity``."""
    text = name.strip().lower()
    text = re.sub(r"[\s_\-]+", " ", text)
    return text


def _slug(name: str) -> str:
    """Alphanumeric-only form, so ``Viral_Pneumonia`` == ``ViralPneumonia``."""
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


_ALIAS_LOOKUP: dict[str, str] = {}
for _canonical, _aliases in _CLASS_ALIASES.items():
    _ALIAS_LOOKUP[_slug(_canonical)] = _canonical
    for _alias in _aliases:
        _ALIAS_LOOKUP[_slug(_alias)] = _canonical


def canonical_class(name: str) -> Optional[str]:
    """Map a folder name onto one of :data:`CLASS_NAMES`, or ``None``."""
    return _ALIAS_LOOKUP.get(_slug(name))


def list_class_images(class_dir: Path) -> list[Path]:
    """Image files of one class folder, supporting both Kaggle layouts.

    ``<class>/images/*.png`` is the layout the official dataset ships; some
    re-uploads flatten it to ``<class>/*.png``. Both are accepted.
    """
    class_dir = Path(class_dir)
    images_sub = class_dir / "images"
    base = images_sub if images_sub.is_dir() else class_dir
    try:
        with os.scandir(base) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
    except OSError:
        return []
    return [
        Path(entry.path)
        for entry in entries
        if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS
    ]


def _class_subdirs(path: str | Path) -> dict[str, Path]:
    """Class folders directly inside ``path``, keyed by canonical class name."""
    found: dict[str, Path] = {}
    try:
        entries = sorted(os.scandir(path), key=lambda entry: entry.name)
    except OSError:
        return found
    for entry in entries:
        try:
            is_dir = entry.is_dir()
        except OSError:
            continue
        if not is_dir:
            continue
        canonical = canonical_class(entry.name)
        if canonical is not None and canonical not in found:
            found[canonical] = Path(entry.path)
    return found


def _has_any_image(class_dir: Path) -> bool:
    images_sub = class_dir / "images"
    base = images_sub if images_sub.is_dir() else class_dir
    try:
        entries = os.scandir(base)
    except OSError:
        return False
    with entries:
        for entry in entries:
            if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS:
                return True
    return False


def _depth_of(path: str, root: str) -> int:
    return os.path.abspath(path).rstrip(os.sep).count(os.sep) - os.path.abspath(root).rstrip(os.sep).count(os.sep)


def resolve_dataset_root(input_root: str = KAGGLE_INPUT, max_depth: int = 4) -> str:
    """Return the directory that directly contains the four class folders.

    Walks ``input_root`` breadth-first to ``max_depth`` and collects every
    directory whose immediate children include all four class folders with at
    least one image each. The shallowest such directory wins; ties break on the
    smallest class (more data). On failure the error lists everything that was
    inspected so the user can see which mount was actually attached.
    """
    root = Path(input_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset root {root} does not exist. On Kaggle attach the COVID-19 "
            "Radiography Database in the notebook's Data panel, or pass "
            "--data-root with the extracted folder."
        )

    root_str = str(root.resolve())
    candidates: list[tuple[int, int, str, dict[str, Path], dict[str, int]]] = []
    inspected: list[tuple[str, dict[str, Path]]] = []

    for dirpath, dirnames, _ in os.walk(root_str):
        depth = _depth_of(dirpath, root_str)
        if depth > max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not name.startswith(".") and name.lower() not in _SKIP_DIRS
        )

        subdirs = _class_subdirs(dirpath)
        if subdirs:
            inspected.append((dirpath, subdirs))
        if len(subdirs) < len(CLASS_NAMES):
            continue

        counts = {name: len(list_class_images(path)) for name, path in subdirs.items()}
        if any(count == 0 for count in counts.values()):
            continue
        candidates.append((min(counts.values()), -depth, dirpath, subdirs, counts))

    if not candidates:
        if inspected:
            listing = "\n".join(
                "    {path}\n      classes found: {found}".format(
                    path=path,
                    found=", ".join(
                        f"{name} ({len(list_class_images(sub))} images)"
                        for name, sub in sorted(subdirs.items())
                    )
                    or "none",
                )
                for path, subdirs in inspected
            )
            detail = (
                f"Searched {root_str} to depth {max_depth}. Directories that looked "
                f"like a dataset but were incomplete:\n{listing}"
            )
        else:
            try:
                top = ", ".join(sorted(os.listdir(root_str))) or "(empty)"
            except OSError:
                top = "(unreadable)"
            detail = (
                f"Searched {root_str} to depth {max_depth} and found no directory "
                f"containing all of {list(CLASS_NAMES)}. Top-level entries: {top}"
            )
        raise FileNotFoundError(
            "Could not locate the chest X-ray dataset root. Expected a directory "
            "directly containing the four class folders "
            f"{list(CLASS_NAMES)} (with <class>/images/*.png or <class>/*.png).\n"
            + detail
        )

    # Shallowest complete candidate wins: the dataset root itself, not a nested
    # copy inside it. item[1] is -depth, so negating it sorts depth ascending.
    candidates.sort(key=lambda item: (-item[1], -item[0]))
    best = candidates[0]
    return best[2]


def class_dirs(dataset_root: str | Path) -> dict[str, Path]:
    """Map every canonical class name to its folder, failing loudly if incomplete."""
    subdirs = _class_subdirs(dataset_root)
    missing = [name for name in CLASS_NAMES if name not in subdirs]
    if missing:
        raise FileNotFoundError(
            f"{dataset_root} is missing class folder(s): {missing}. "
            f"Found: {sorted(subdirs)}"
        )
    empty = [name for name in CLASS_NAMES if not _has_any_image(subdirs[name])]
    if empty:
        raise FileNotFoundError(
            f"Class folder(s) {empty} under {dataset_root} contain no images. "
            "Expected <class>/images/*.png or <class>/*.png."
        )
    return {name: subdirs[name] for name in CLASS_NAMES}


def _read_readme_text(dataset_root: Optional[str | Path]) -> str:
    if dataset_root is None:
        return ""
    root = Path(dataset_root)
    for candidate in (
        root / "README.md",
        root / "README.txt",
        root / "readme.md",
        root.parent / "README.md",
    ):
        try:
            if candidate.is_file():
                with open(candidate, encoding="utf-8", errors="replace") as handle:
                    return handle.read(64_000).lower()
        except OSError:
            continue
    return ""


def _pool_from_text(text: str, class_name: str) -> Optional[str]:
    """Best-effort provenance pool for one class, read from the dataset README."""
    if not text:
        return None
    token = _slug(class_name)
    for line in text.splitlines():
        if token and token not in _slug(line):
            continue
        if "viral" in line:
            return "viral"
        if "rsna" in line or "chestx-ray14" in line or "chestxray14" in line or "nih" in line:
            return "rsna"
        if "covid-chestxray" in line or "cohen" in line or "radiography database" in line:
            return "covid_mixed"
    return None


def source_pool_for(class_name: str, dataset_root: Optional[str | Path] = None) -> str:
    """Provenance pool of a class, from the README when present else the folder name.

    The pool is bookkeeping for the leakage audit, not a label: the COVID folder
    of the Radiography Database mixes several source corpora, hence
    ``covid_mixed``. Unknown classes fall back to their own slug.
    """
    pool = _pool_from_text(_read_readme_text(dataset_root), class_name)
    if pool:
        return pool
    return SOURCE_POOL_DEFAULTS.get(class_name, _slug(class_name) or "unknown")


@dataclass
class LungConfig:
    """Full run configuration, serialised into the resume payload."""

    # ── data ─────────────────────────────────────────────────────────────
    data_root: Optional[str] = None
    output_dir: str = os.path.join(KAGGLE_WORKING, "lung_v2s_run")
    seed: int = 42

    # ── split ────────────────────────────────────────────────────────────
    val_frac: float = 0.15
    test_frac: float = 0.15
    phash_hamming: int = 5

    # ── optimisation (contract values) ───────────────────────────────────
    epochs: int = 100
    patience: int = 15
    monitor: str = "val_macro_f1"
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    pct_start: float = 0.25
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    amp: bool = True
    ema_decay: float = 0.999

    # ── model ────────────────────────────────────────────────────────────
    model_tag: str = MODEL_TAG
    num_classes: int = NUM_CLASSES
    drop_path_rate: float = 0.2
    head_dropout: float = 0.3
    train_size: int = 300
    infer_size: int = 300
    infer_size_alt: int = 384

    # ── augmentation ─────────────────────────────────────────────────────
    rand_augment: str = "rand-m9-mstd0.5-inc1"
    rrc_scale: tuple[float, float] = (0.8, 1.0)
    rrc_ratio: tuple[float, float] = (0.75, 1.3333333333333333)
    hflip_p: float = 0.5
    rotation_deg: float = 10.0
    minority_class: str = MINORITY_CLASS
    minority_extra_draw_prob: float = 0.5
    minority_rand_augment: str = "rand-m11-mstd0.5-inc1"

    # ── mixing ───────────────────────────────────────────────────────────
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mixup_prob: float = 0.5
    mixup_off_last_epochs: int = 10

    # ── packaging ────────────────────────────────────────────────────────
    hash_workers: int = 8
    zip_name: str = "output.zip"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LungConfig:
        names = {field.name for field in fields(cls)}
        data = {key: value for key, value in payload.items() if key in names}
        for key in ("rrc_scale", "rrc_ratio"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        return cls(**data)


__all__ = [
    "CLASS_NAMES",
    "CLASS_TO_INDEX",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IMAGE_EXTENSIONS",
    "KAGGLE_INPUT",
    "KAGGLE_WORKING",
    "MINORITY_CLASS",
    "MODEL_TAG",
    "NUM_CLASSES",
    "SOURCE_POOL_DEFAULTS",
    "LungConfig",
    "canonical_class",
    "class_dirs",
    "list_class_images",
    "normalise_name",
    "resolve_dataset_root",
    "source_pool_for",
]
