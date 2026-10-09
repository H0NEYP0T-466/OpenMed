"""Image contract shared by training and serving.

Training and inference MUST build the same tensor from the same file, or the
trained weights are served inputs they never saw. Both sides therefore call the
functions in this module and nothing else:

1. decode to 8-bit single channel (``cv2`` grayscale decode — the OpenMed JPEGs
   store R == G == B, the BTSC PNGs are single channel, so nothing is lost);
2. resize so the longest side equals ``size`` (``INTER_AREA`` when shrinking,
   ``INTER_LINEAR`` when enlarging) and zero-pad bottom/right to ``size x size``;
3. per-image intensity normalisation computed over the head region only.

Masks follow the same geometry (area resampling, then a 0.5 threshold).
``PREPROCESS_VERSION`` is stored in every checkpoint; serving refuses a
checkpoint whose version it does not implement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import cv2
import numpy as np

PREPROCESS_VERSION = 1
DEFAULT_IMAGE_SIZE = 256
MASK_THRESHOLD_U8 = 128

_FOREGROUND_FLOOR = 5.0
_MIN_FOREGROUND_FRACTION = 0.02
_CLIP_PERCENTILES = (0.5, 99.5)


@dataclass(frozen=True)
class Geometry:
    """How an original image maps onto the square model canvas."""

    original_h: int
    original_w: int
    scaled_h: int
    scaled_w: int
    size: int

    @property
    def scale(self) -> float:
        return self.scaled_h / self.original_h


def preprocess_spec(size: int = DEFAULT_IMAGE_SIZE) -> dict[str, Any]:
    """The contract, as data, for embedding in a checkpoint."""
    return {
        "version": PREPROCESS_VERSION,
        "image_size": int(size),
        "channels": 1,
        "decode": "cv2 grayscale",
        "resize": "longest side -> size; INTER_AREA down / INTER_LINEAR up",
        "pad": "zeros, bottom and right",
        "normalise": {
            "foreground": f"pixel > max({_FOREGROUND_FLOOR}, 0) over the canvas",
            "min_foreground_fraction": _MIN_FOREGROUND_FRACTION,
            "clip_percentiles": list(_CLIP_PERCENTILES),
            "scheme": "clip to percentiles of foreground, then z-score over foreground",
        },
        "mask": f"area resample, binarise at >= {MASK_THRESHOLD_U8} before resize, 0.5 after",
    }


def decode_gray(data: bytes) -> np.ndarray:
    """Decode encoded image bytes to a uint8 grayscale array (same decoder as files)."""
    buffer = np.frombuffer(data, dtype=np.uint8)
    gray = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Image could not be decoded.")
    return gray


def read_gray(path: str) -> np.ndarray:
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Image could not be read: {path}")
    return gray


def to_gray_uint8(image: Union[np.ndarray, Any]) -> np.ndarray:
    """Coerce an array or PIL image to uint8 grayscale.

    Only for callers that already hold decoded pixels. Anything starting from
    bytes or a path must use :func:`decode_gray` / :func:`read_gray` so the
    decoder matches training exactly.
    """
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = array[:, :, :3]
        if array.shape[2] == 3:
            array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            array = array[:, :, 0]
    if array.dtype != np.uint8:
        peak = float(array.max()) if array.size else 0.0
        scale = 255.0 / peak if peak > 255.0 else 1.0
        array = np.clip(array.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def letterbox_geometry(height: int, width: int, size: int) -> Geometry:
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size {height}x{width}.")
    scale = size / max(height, width)
    scaled_h = max(1, min(size, int(round(height * scale))))
    scaled_w = max(1, min(size, int(round(width * scale))))
    return Geometry(height, width, scaled_h, scaled_w, size)


def _resize(array: np.ndarray, geom: Geometry, *, area_only: bool = False) -> np.ndarray:
    shrinking = geom.scaled_h < geom.original_h or geom.scaled_w < geom.original_w
    interpolation = cv2.INTER_AREA if (shrinking or area_only) else cv2.INTER_LINEAR
    if geom.scaled_h == geom.original_h and geom.scaled_w == geom.original_w:
        return array
    return cv2.resize(array, (geom.scaled_w, geom.scaled_h), interpolation=interpolation)


def to_canvas(gray: np.ndarray, size: int = DEFAULT_IMAGE_SIZE) -> tuple[np.ndarray, Geometry]:
    """Resize and zero-pad a grayscale image to ``size x size`` uint8."""
    if gray.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale array, got shape {gray.shape}.")
    geom = letterbox_geometry(gray.shape[0], gray.shape[1], size)
    resized = _resize(gray, geom)
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[: geom.scaled_h, : geom.scaled_w] = resized
    return canvas, geom


def mask_to_canvas(mask: np.ndarray, size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Resize a mask with the image geometry; returns uint8 in {0, 1}."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    binary = (mask >= MASK_THRESHOLD_U8).astype(np.float32)
    geom = letterbox_geometry(binary.shape[0], binary.shape[1], size)
    resized = _resize(binary, geom, area_only=True)
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[: geom.scaled_h, : geom.scaled_w] = (resized >= 0.5).astype(np.uint8)
    return canvas


def normalise(canvas: np.ndarray) -> np.ndarray:
    """Per-image robust z-score over the head region. Returns float32 ``(H, W)``."""
    x = canvas.astype(np.float32)
    if float(x.max()) <= 0.0:
        return np.zeros_like(x)

    foreground = x > _FOREGROUND_FLOOR
    if foreground.mean() < _MIN_FOREGROUND_FRACTION:
        foreground = np.ones_like(foreground, dtype=bool)

    values = x[foreground]
    low, high = np.percentile(values, _CLIP_PERCENTILES)
    if high <= low:
        high = low + 1.0
    x = np.clip(x, low, high)
    values = x[foreground]
    mean = float(values.mean())
    std = float(values.std())
    return ((x - mean) / max(std, 1e-3)).astype(np.float32)


def preprocess_gray(
    gray: np.ndarray, size: int = DEFAULT_IMAGE_SIZE
) -> tuple[np.ndarray, Geometry]:
    """Grayscale uint8 -> normalised float32 ``(1, size, size)`` plus geometry."""
    canvas, geom = to_canvas(gray, size)
    return normalise(canvas)[None, :, :], geom


def canvas_to_original(
    plane: np.ndarray, geom: Geometry, *, interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """Crop away the padding and resample a canvas-space plane to the original size."""
    cropped = plane[: geom.scaled_h, : geom.scaled_w]
    if geom.scaled_h == geom.original_h and geom.scaled_w == geom.original_w:
        return np.ascontiguousarray(cropped)
    return cv2.resize(
        cropped, (geom.original_w, geom.original_h), interpolation=interpolation
    )
