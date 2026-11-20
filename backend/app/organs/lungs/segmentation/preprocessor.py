"""Image contract shared by training and serving.

Training and inference MUST build the same tensor from the same file, or the
trained weights are served inputs they never saw. Both sides therefore call the
functions in this module and nothing else; nothing here is parameterised from
outside, and the version is stamped into every checkpoint.

Contract (``PREPROCESS_VERSION`` 1)

1. decode to 8-bit single channel with the cv2 grayscale decoder (COVID-QU-Ex
   PNGs are grayscale-safe, verified by hygiene check H04);
2. resize to ``256 x 256`` (``INTER_AREA`` when shrinking, ``INTER_LINEAR`` when
   enlarging); the dataset is natively 256 x 256 so training never resamples;
3. ``uint8 / 255``, replicated to 3 channels, ImageNet mean/std.

No CLAHE, no histogram equalisation, no intensity transform that exists at
inference but not in training. Intensity augmentation lives in the training
package only and is applied to the uint8 canvas *before* this contract.

Head B (COVID opacity) runs on a crop around the lung field, resized back to
256 x 256. The crop geometry is part of the contract too, because a different
margin at serving time is a different input distribution.

The tensor construction is a single torch function used by both sides, so
train/serve parity holds by construction and the parity test is a real check,
not a tautology between two copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

PREPROCESS_VERSION = 1
IMAGE_SIZE = 256
MASK_THRESHOLD_U8 = 128

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Lung-field crop for head B. Margin is a fraction of the box side added on each
# side; a box narrower than ``CROP_MIN_SIDE`` pixels is grown around its centre.
CROP_MARGIN = 0.06
CROP_MIN_SIDE = 64


def preprocess_spec() -> dict[str, Any]:
    """The contract, as data, for embedding in a checkpoint."""
    return {
        "version": PREPROCESS_VERSION,
        "image_size": IMAGE_SIZE,
        "channels_in": 1,
        "channels_model": 3,
        "decode": "cv2 grayscale",
        "resize": "to 256x256; INTER_AREA down / INTER_LINEAR up; native 256 is untouched",
        "clahe": False,
        "normalise": {
            "scale": "uint8 / 255",
            "replicate_to_channels": 3,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "mask": f"binarise at >= {MASK_THRESHOLD_U8}; resample INTER_LINEAR then >= 0.5",
        "opacity_crop": {
            "source": "bounding box of the lung field (ground truth in training, predicted at serving)",
            "margin_fraction": CROP_MARGIN,
            "min_side_px": CROP_MIN_SIDE,
            "resize": "INTER_LINEAR to 256x256 (aspect ratio is not preserved)",
        },
    }


# --------------------------------------------------------------------------- decode


def decode_gray(data: bytes) -> np.ndarray:
    """Decode encoded image bytes to uint8 grayscale (same decoder as files)."""
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


def read_mask(path: str) -> np.ndarray:
    """Read a binary mask as uint8 ``{0, 1}`` at its native resolution."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Mask could not be read: {path}")
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    return (raw >= MASK_THRESHOLD_U8).astype(np.uint8)


def to_gray_uint8(image: Any) -> np.ndarray:
    """Coerce an array or PIL image to uint8 grayscale.

    Only for callers that already hold decoded pixels. Anything starting from
    bytes or a path must use :func:`decode_gray` / :func:`read_gray`.
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


# -------------------------------------------------------------------------- geometry


@dataclass(frozen=True)
class Geometry:
    """How an original image maps onto the 256 x 256 model canvas."""

    original_h: int
    original_w: int
    size: int = IMAGE_SIZE

    @property
    def is_native(self) -> bool:
        return self.original_h == self.size and self.original_w == self.size


def to_canvas(gray: np.ndarray, size: int = IMAGE_SIZE) -> tuple[np.ndarray, Geometry]:
    """Resize a grayscale image to ``size x size`` uint8 (no-op when already native)."""
    if gray.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale array, got shape {gray.shape}.")
    height, width = gray.shape
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size {height}x{width}.")
    geom = Geometry(height, width, size)
    if geom.is_native:
        return np.ascontiguousarray(gray), geom
    shrinking = height > size or width > size
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(gray, (size, size), interpolation=interpolation), geom


def mask_to_canvas(mask: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Resample a ``{0, 1}`` mask to the canvas; returns uint8 ``{0, 1}``."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    binary = (mask > 0).astype(np.float32)
    if binary.shape == (size, size):
        return binary.astype(np.uint8)
    resized = cv2.resize(binary, (size, size), interpolation=cv2.INTER_LINEAR)
    return (resized >= 0.5).astype(np.uint8)


def canvas_to_original(
    plane: np.ndarray, geom: Geometry, *, interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """Resample a canvas-space plane back to the original resolution."""
    if geom.is_native:
        return np.ascontiguousarray(plane)
    return cv2.resize(plane, (geom.original_w, geom.original_h), interpolation=interpolation)


# --------------------------------------------------------------------------- tensor

_MEAN = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
_STD = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)


def normalise_u8(canvases: torch.Tensor) -> torch.Tensor:
    """``(B, H, W)`` uint8 canvases -> ``(B, 3, H, W)`` float32 model input.

    The single implementation of the tensor contract. Training calls it on the
    accelerator with the batch it just moved; serving calls it on one image.
    """
    if canvases.ndim != 3:
        raise ValueError(f"Expected (B, H, W) canvases, got shape {tuple(canvases.shape)}.")
    x = canvases.to(torch.float32).div_(255.0).unsqueeze(1).expand(-1, 3, -1, -1)
    mean = _MEAN.to(x.device)
    std = _STD.to(x.device)
    return ((x - mean) / std).contiguous()


def canvases_to_tensor(canvases: list[np.ndarray] | np.ndarray) -> torch.Tensor:
    """CPU convenience wrapper around :func:`normalise_u8`."""
    stacked = np.stack(list(canvases)) if isinstance(canvases, list) else np.asarray(canvases)
    return normalise_u8(torch.from_numpy(np.ascontiguousarray(stacked, dtype=np.uint8)))


# ------------------------------------------------------------------------ lung crop


def lung_crop_box(
    lung_mask: np.ndarray,
    margin: float = CROP_MARGIN,
    min_side: int = CROP_MIN_SIDE,
) -> tuple[int, int, int, int] | None:
    """``(x0, y0, x1, y1)`` (exclusive end) around the lung field, or ``None`` if empty."""
    ys, xs = np.nonzero(lung_mask)
    if xs.size == 0:
        return None
    height, width = lung_mask.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return grow_box((x0, y0, x1, y1), (height, width), margin, min_side)


def grow_box(
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
    margin: float,
    min_side: int,
) -> tuple[int, int, int, int]:
    """Add a fractional margin, enforce a minimum side, clamp to the frame."""
    height, width = shape
    x0, y0, x1, y1 = box
    pad_x = round((x1 - x0) * margin)
    pad_y = round((y1 - y0) * margin)
    x0, x1 = x0 - pad_x, x1 + pad_x
    y0, y1 = y0 - pad_y, y1 + pad_y
    if x1 - x0 < min_side:
        cx = (x0 + x1) // 2
        x0, x1 = cx - min_side // 2, cx - min_side // 2 + min_side
    if y1 - y0 < min_side:
        cy = (y0 + y1) // 2
        y0, y1 = cy - min_side // 2, cy - min_side // 2 + min_side
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, width, height
    return x0, y0, x1, y1


def full_box(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    return 0, 0, int(shape[1]), int(shape[0])


def crop_resize_image(
    canvas: np.ndarray, box: tuple[int, int, int, int], size: int = IMAGE_SIZE
) -> np.ndarray:
    x0, y0, x1, y1 = box
    return cv2.resize(canvas[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_LINEAR)


def crop_resize_mask(
    mask: np.ndarray, box: tuple[int, int, int, int], size: int = IMAGE_SIZE
) -> np.ndarray:
    x0, y0, x1, y1 = box
    window = (mask[y0:y1, x0:x1] > 0).astype(np.float32)
    resized = cv2.resize(window, (size, size), interpolation=cv2.INTER_LINEAR)
    return (resized >= 0.5).astype(np.uint8)


def uncrop_plane(
    plane: np.ndarray, box: tuple[int, int, int, int], canvas_shape: tuple[int, int]
) -> np.ndarray:
    """Paste a crop-space float plane back onto the canvas (zeros outside the box)."""
    x0, y0, x1, y1 = box
    out = np.zeros(canvas_shape, dtype=np.float32)
    out[y0:y1, x0:x1] = cv2.resize(
        plane.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR
    )
    return out
