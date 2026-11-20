"""Label space, supervision scope and the display merge rule.

This file is the only place that says *what a pixel value means* and *which
images supervise which head*. Training, evaluation and serving all import it.

Supervision scope (decided, see README "Supervision scope")
    Head A, lung field      every image
    Head B, COVID opacity   COVID-19 (positive) and Normal (true negative) images
                            that belong to the infection tree. Non-COVID is
                            excluded from the loss entirely: its all-black
                            infection mask is an *annotation negative* (the
                            annotators only delineated COVID-type opacities and
                            never drew bacterial consolidation), not a true
                            negative. Images outside the infection tree carry no
                            annotation at all.

The flag is **provenance**, never pixel content. ``mask.sum() > 0`` would drop
Normal together with Non-COVID and leave head B with positives only, a head that
predicts opacity everywhere and reports an excellent training Dice.
"""

from __future__ import annotations

from typing import Any

import numpy as np

TASK_NAME = "COVID opacity segmentation"

HEADS: tuple[str, ...] = ("lung", "opacity")

CLASS_COVID = "COVID-19"
CLASS_NON_COVID = "Non-COVID"
CLASS_NORMAL = "Normal"
CLASSES: tuple[str, ...] = (CLASS_COVID, CLASS_NON_COVID, CLASS_NORMAL)
CLASS_IDS: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}

# Classes whose infection mask is valid head-B supervision when the image is in
# the infection tree.
OPACITY_SUPERVISED_CLASSES: frozenset[str] = frozenset({CLASS_COVID, CLASS_NORMAL})

MERGE_BACKGROUND = 0
MERGE_LUNG = 1
MERGE_OPACITY = 2

LEGEND: tuple[dict[str, Any], ...] = (
    {"value": MERGE_BACKGROUND, "name": "Background"},
    {"value": MERGE_LUNG, "name": "Lung field"},
    {"value": MERGE_OPACITY, "name": "COVID-type opacity (within the lung field)"},
)

SCOPE_NOTICE = (
    "Delineates COVID-type opacities only, as annotated in COVID-QU-Ex. It does not "
    "delineate other pneumonia (for example bacterial consolidation) and is not a "
    "diagnosis. Research use only."
)

NON_COVID_NOTE = (
    "Out-of-scope behaviour, not a score: Non-COVID infection masks are annotation "
    "negatives (bacterial consolidation was never drawn), so no Dice is computed."
)


def normalise_class(name: str) -> str:
    """Map a directory or filename spelling to the canonical class name."""
    token = name.strip().lower().replace("_", "-").replace(" ", "-")
    if token.startswith("covid"):
        return CLASS_COVID
    if token.startswith("non"):
        return CLASS_NON_COVID
    if token.startswith("normal"):
        return CLASS_NORMAL
    raise ValueError(f"Unrecognised class name {name!r}")


def opacity_supervised(class_name: str, in_infection_tree: bool) -> bool:
    """Provenance flag for head B.

    True for COVID-19 and Normal images that belong to the infection tree; false
    for Non-COVID and for every image outside the infection tree. Deliberately
    independent of any mask.
    """
    return bool(in_infection_tree) and normalise_class(class_name) in OPACITY_SUPERVISED_CLASSES


def merge_label_map(lung_mask: np.ndarray, opacity_mask: np.ndarray | None) -> np.ndarray:
    """Render one 0/1/2 map: background, lung field only, opacity inside the lung.

    A rendering rule for a single output image, not a softmax over three classes.
    Opacity pixels outside the lung field are discarded, so a value of 2 always
    implies a lung pixel.
    """
    lung = np.asarray(lung_mask) > 0
    out = np.zeros(lung.shape, dtype=np.uint8)
    out[lung] = MERGE_LUNG
    if opacity_mask is not None:
        opacity = (np.asarray(opacity_mask) > 0) & lung
        out[opacity] = MERGE_OPACITY
    return out
