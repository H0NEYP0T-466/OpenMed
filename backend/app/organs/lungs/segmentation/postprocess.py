"""Post-processing, shared by the validation search and by serving.

The same functions tune on validation and run in the application, so the tuned
settings are the settings that ship.

Lung field   threshold, drop specks, keep the ``k`` largest components, fill holes.
             A lung is always present, so if filtering would remove every
             component the largest one is kept.
Opacity      threshold, drop specks, clip to the lung field. An empty answer is
             legitimate (that is the point on Normal lungs), so there is no
             "keep the largest" fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class LungPost:
    threshold: float = 0.5
    min_area_frac: float = 0.0
    keep_largest_k: int = 2          # 0 keeps every component
    fill_holes: bool = True
    tta_hflip: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LungPost:
        return cls(
            threshold=float(payload.get("threshold", 0.5)),
            min_area_frac=float(payload.get("min_area_frac", 0.0)),
            keep_largest_k=int(payload.get("keep_largest_k", 2)),
            fill_holes=bool(payload.get("fill_holes", True)),
            tta_hflip=bool(payload.get("tta_hflip", False)),
        )


@dataclass(frozen=True)
class OpacityPost:
    threshold: float = 0.5
    min_area_frac: float = 0.0
    tta_hflip: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OpacityPost:
        return cls(
            threshold=float(payload.get("threshold", 0.5)),
            min_area_frac=float(payload.get("min_area_frac", 0.0)),
            tta_hflip=bool(payload.get("tta_hflip", False)),
        )


def component_areas(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """``(count, labels, areas)`` for the foreground components of a binary mask."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    return count - 1, labels, stats[1:, cv2.CC_STAT_AREA]


def drop_specks(mask: np.ndarray, min_area_frac: float, *, keep_if_all_removed: bool) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if min_area_frac <= 0.0 or not mask.any():
        return mask
    minimum = max(1, round(min_area_frac * mask.size))
    count, labels, areas = component_areas(mask)
    if count == 0:
        return mask
    keep = [i + 1 for i, area in enumerate(areas) if area >= minimum]
    if not keep:
        if not keep_if_all_removed:
            return np.zeros_like(mask)
        keep = [1 + int(np.argmax(areas))]
    return np.isin(labels, keep).astype(np.uint8)


def keep_largest(mask: np.ndarray, k: int) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if k <= 0 or not mask.any():
        return mask
    count, labels, areas = component_areas(mask)
    if count <= k:
        return mask
    order = np.argsort(areas)[::-1][:k]
    return np.isin(labels, [int(i) + 1 for i in order]).astype(np.uint8)


def postprocess_lung(probability: np.ndarray, post: LungPost) -> np.ndarray:
    mask = (probability >= post.threshold).astype(np.uint8)
    mask = drop_specks(mask, post.min_area_frac, keep_if_all_removed=True)
    mask = keep_largest(mask, post.keep_largest_k)
    if post.fill_holes and mask.any():
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    return mask


def postprocess_opacity(
    probability: np.ndarray, lung_mask: np.ndarray | None, post: OpacityPost
) -> np.ndarray:
    mask = (probability >= post.threshold).astype(np.uint8)
    if lung_mask is not None:
        mask = mask & (lung_mask > 0).astype(np.uint8)
    return drop_specks(mask, post.min_area_frac, keep_if_all_removed=False)
