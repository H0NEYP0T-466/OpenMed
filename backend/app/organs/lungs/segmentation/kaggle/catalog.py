"""Dataset catalog: one record per lung-tree image, infection attachments by provenance.

The infection tree is a strict filename subset of the lung tree (verified on disk:
all 5,826 infection files exist in the lung tree under the same class and split,
and are pixel-identical to their twins). So the lung tree is the master list and
a record carries the infection-tree files of its twin when it has one. That is
what lets one index over every image hold a per-sample supervision flag.

The supervision flag comes from ``labels.opacity_supervised`` (class + tree
membership). It is never derived from mask pixels.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from ..labels import CLASSES, opacity_supervised
from .config import (
    IMAGES_DIR,
    INFECTION_MASK_DIR,
    LUNG_MASK_DIR,
    SPLITS,
    TrainConfig,
    class_dirs,
    split_dir_names,
)

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

_STEM_RE = re.compile(r"^(covid|non[\s_\-]?covid|normal)[\s_\-]*\(?\s*(\d+)\s*\)?$", re.IGNORECASE)


# The COVID-19 class mixes two naming schemes: ``covid_<n>`` and BIDS-style
# ``sub-S<subject>_ses-E<session>_...`` files from a different source. Both are valid
# stems; the BIDS form carries a subject id, which is a group key in its own right.
_BIDS_RE = re.compile(r"^sub-(s?\d+)_ses-(e?\d+)", re.IGNORECASE)


def parse_bids_subject(filename: str) -> str | None:
    match = _BIDS_RE.match(os.path.splitext(filename)[0].strip())
    return match.group(1).upper() if match else None


def parse_stem(filename: str, cls: str | None = None) -> tuple[str, int] | None:
    """``(class, number)`` for ``covid_<n>`` style names, or ``None`` when it does not parse.

    BIDS-style names parse to ``(cls, 0)`` when the caller supplies the directory class,
    because the filename itself names no class.
    """
    stem = os.path.splitext(filename)[0].strip()
    if cls is not None and parse_bids_subject(filename) is not None:
        return cls, 0
    match = _STEM_RE.match(stem)
    if not match:
        return None
    token = match.group(1).lower().replace("_", "").replace("-", "").replace(" ", "")
    cls = "COVID-19" if token == "covid" else "Non-COVID" if token == "noncovid" else "Normal"
    return cls, int(match.group(2))


@dataclass
class Record:
    uid: int
    filename: str
    cls: str
    shipped_split: str
    image_path: str
    lung_mask_path: str
    in_infection_tree: bool = False
    infection_mask_path: str | None = None
    infection_image_path: str | None = None
    infection_lung_mask_path: str | None = None

    @property
    def stem(self) -> str:
        return os.path.splitext(self.filename)[0]

    @property
    def opacity_supervised(self) -> bool:
        return opacity_supervised(self.cls, self.in_infection_tree)

    @property
    def tree(self) -> str:
        return "lung+infection" if self.in_infection_tree else "lung_only"

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["opacity_supervised"] = self.opacity_supervised
        return out


@dataclass
class Catalog:
    records: list[Record]
    pairing: dict[str, list[str]]       # problem kind -> offending paths
    orphan_rows: list[dict[str, str]]   # rows for dropped_files.csv
    roots: dict[str, str]
    tree_counts: dict[str, dict[str, dict[str, int]]]  # tree -> split -> class -> n


def _scan(directory: str) -> dict[str, str]:
    """``{filename: path}`` for image-like files in ``directory`` (empty if missing)."""
    out: dict[str, str] = {}
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(IMG_EXT):
                    out[entry.name] = entry.path
    except OSError:
        pass
    return out


def build_catalog(roots: dict[str, str], cfg: TrainConfig) -> Catalog:
    lung_root, infection_root = roots["lung"], roots["infection"]
    pairing: dict[str, list[str]] = defaultdict(list)
    orphan_rows: list[dict[str, str]] = []
    counts: dict[str, dict[str, dict[str, int]]] = {"lung": {}, "infection": {}}

    def row(path: str, tree: str, split: str, cls: str, reason: str, detail: str = "") -> dict[str, str]:
        return {"path": path, "tree": tree, "split": split, "class": cls, "reason": reason,
                "detail": detail}

    # -- lung tree: the master list ----------------------------------------------------
    lung_dirs = split_dir_names(lung_root)
    lung_records: dict[tuple[str, str], Record] = {}
    for split in SPLITS:
        counts["lung"][split] = {}
        for cls, cdir in class_dirs(lung_root, lung_dirs[split]).items():
            images = _scan(os.path.join(cdir, IMAGES_DIR))
            masks = _scan(os.path.join(cdir, LUNG_MASK_DIR))
            names = sorted(set(images) & set(masks))
            if cfg.limit_per_class:
                names = names[: cfg.limit_per_class]
            counts["lung"][split][cls] = len(names)
            for name in sorted(set(images) - set(masks)):
                pairing["lung_image_without_mask"].append(images[name])
                orphan_rows.append(row(images[name], "lung", split, cls, "orphan_image_no_mask"))
            for name in sorted(set(masks) - set(images)):
                pairing["lung_mask_without_image"].append(masks[name])
                orphan_rows.append(row(masks[name], "lung", split, cls, "orphan_mask_no_image"))
            for name in names:
                lung_records[(cls, name)] = Record(
                    uid=-1, filename=name, cls=cls, shipped_split=split,
                    image_path=images[name], lung_mask_path=masks[name],
                )

    # -- infection tree: attach by provenance ------------------------------------------
    inf_dirs = split_dir_names(infection_root)
    for split in SPLITS:
        counts["infection"][split] = {}
        for cls, cdir in class_dirs(infection_root, inf_dirs[split]).items():
            images = _scan(os.path.join(cdir, IMAGES_DIR))
            lung_masks = _scan(os.path.join(cdir, LUNG_MASK_DIR))
            inf_masks = _scan(os.path.join(cdir, INFECTION_MASK_DIR))
            names = sorted(set(images) & set(lung_masks) & set(inf_masks))
            if cfg.limit_per_class:
                names = [n for n in names if (cls, n) in lung_records]
            counts["infection"][split][cls] = len(names)
            for kind, left, right in (
                ("infection_image_without_masks", images, set(lung_masks) & set(inf_masks)),
                ("infection_lung_mask_without_image", lung_masks, set(images)),
                ("infection_mask_without_image", inf_masks, set(images)),
            ):
                for name in sorted(set(left) - right):
                    pairing[kind].append(left[name])
                    orphan_rows.append(row(left[name], "infection", split, cls, f"orphan_{kind}"))
            for name in names:
                twin = lung_records.get((cls, name))
                if twin is None:
                    pairing["infection_without_lung_twin"].append(images[name])
                    orphan_rows.append(row(images[name], "infection", split, cls,
                                           "infection_file_without_lung_tree_twin"))
                    continue
                if twin.shipped_split != split:
                    pairing["infection_twin_in_other_split"].append(images[name])
                twin.in_infection_tree = True
                twin.infection_image_path = images[name]
                twin.infection_lung_mask_path = lung_masks[name]
                twin.infection_mask_path = inf_masks[name]

    records = sorted(lung_records.values(), key=lambda r: (SPLITS.index(r.shipped_split),
                                                           CLASSES.index(r.cls), r.filename))
    for uid, record in enumerate(records):
        record.uid = uid
    return Catalog(records=records, pairing=dict(pairing), orphan_rows=orphan_rows,
                   roots=dict(roots), tree_counts=counts)
