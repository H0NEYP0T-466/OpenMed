"""Canonical label space for the 9-class brain tumour classifier.

A checkpoint and the name list used to interpret its logits must agree exactly.
The label space is therefore persisted next to the weights at training time and
re-verified at load time; a disagreement raises instead of silently relabelling.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

MRI_SEQUENCES = ("T1", "T1C+", "T2")

CLASS_NAMES: tuple[str, ...] = (
    "Germ Cell Tumors",
    "Gliomas",
    "Medulloblastoma",
    "Meningothelial Tumors",
    "Mesenchymal (Non-Meningothelial Tumors)",
    "Mixed Neuronal and Neuronal-Glial Tumors",
    "Normal",
    "Pituitary",
    "Schwannoma",
)


LABEL_SPACE_FILENAME = "label_space.json"


class LabelSpaceError(RuntimeError):
    """Raised when a checkpoint's head cannot be mapped onto a trusted name list."""


@dataclass(frozen=True)
class LabelSpace:
    class_names: tuple[str, ...]
    model_tag: str = ""
    source: str = "unknown"
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def index_of(self, class_name: str) -> int:
        return self.class_names.index(class_name)


def label_space_path_for(checkpoint_path: str) -> str:
    stem, _ = os.path.splitext(checkpoint_path)
    return f"{stem}.{LABEL_SPACE_FILENAME}"


def save_label_space(
    checkpoint_path: str,
    class_names: Any,
    model_tag: str = "",
    metrics: Optional[dict[str, Any]] = None,
) -> str:
    target = label_space_path_for(checkpoint_path)
    payload = {
        "class_names": list(class_names),
        "num_classes": len(list(class_names)),
        "model_tag": model_tag,
        "metrics": metrics or {},
    }
    with open(target, "w") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Label space written to %s", target)
    return target


def read_label_space(checkpoint_path: str) -> Optional[LabelSpace]:
    target = label_space_path_for(checkpoint_path)
    if not checkpoint_path or not os.path.isfile(target):
        return None
    try:
        with open(target) as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabelSpaceError(f"Label file {target} is unreadable: {exc}") from exc

    names = tuple(payload.get("class_names", ()))
    if not names:
        raise LabelSpaceError(f"Label file {target} lists no class names.")
    return LabelSpace(
        class_names=names,
        model_tag=payload.get("model_tag", ""),
        source=f"sidecar:{target}",
        metrics=payload.get("metrics", {}),
    )


def split_class_name(class_name: str) -> tuple[str, str]:
    """Separate a class name into (tumour type, MRI sequence)."""
    base, _, tail = class_name.rpartition(" ")
    if base and tail in MRI_SEQUENCES:
        return base, tail
    return class_name, "Unknown"


def tumor_type_of(class_name: str) -> str:
    return split_class_name(class_name)[0]


def sequence_of(class_name: str) -> str:
    return split_class_name(class_name)[1]


def resolve_label_space(
    checkpoint_path: Optional[str],
    detected_classes: Optional[int],
    fallback: Optional[Any] = None,
) -> LabelSpace:
    """Bind a checkpoint's logit width to a trustworthy list of class names.

    Order of preference: the sidecar label file, then the in-repo canonical list
    when its width matches. Anything else raises, because guessing the mapping is
    how a 39-class head came to be read as 42 names.
    """
    sidecar = read_label_space(checkpoint_path) if checkpoint_path else None
    if sidecar is not None:
        if detected_classes is not None and sidecar.num_classes != detected_classes:
            raise LabelSpaceError(
                f"{checkpoint_path} has a {detected_classes}-class head but "
                f"{sidecar.source} declares {sidecar.num_classes} classes. "
                "Retrain, or regenerate the label file from the training run."
            )
        return sidecar

    names = tuple(fallback if fallback is not None else CLASS_NAMES)
    if detected_classes is None:
        return LabelSpace(class_names=names, source="canonical-unverified")

    if detected_classes == len(names):
        return LabelSpace(class_names=names, source="canonical")

    raise LabelSpaceError(
        f"Checkpoint '{checkpoint_path or '<none>'}' exposes "
        f"{detected_classes} logits, which does not match the {len(names)} names "
        "in the brain label space. Refusing to guess the mapping: copy "
        f"'{LABEL_SPACE_FILENAME}' next to the checkpoint, or retrain on the "
        f"current {len(names)}-class dataset."
    )
