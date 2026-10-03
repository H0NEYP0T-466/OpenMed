"""Image preprocessing for LiteMedSAM segmentation.

LiteMedSAM expects 256×256 RGB images normalised to [0, 1].  The
preprocessing pipeline mirrors the official inference code:

1. Resize longest side to 256, preserving aspect ratio.
2. Normalise intensities to [0, 1].
3. Zero-pad the shorter side (right/bottom) to reach 256×256.
4. Arrange as a ``(B, 3, 256, 256)`` float32 tensor.

All geometry transforms (box coordinates, output mask crop) are tracked
so that results can be mapped back to the original image dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MEDSAM_INPUT_SIZE: int = 256


@dataclass(frozen=True)
class PreprocessResult:
    """Carries the tensor and the geometry needed to undo padding/scaling."""

    tensor: torch.Tensor          # (1, 3, 256, 256)
    original_hw: tuple[int, int]  # (H, W) of the original image
    resized_hw: tuple[int, int]   # (H, W) after resize, before padding
    scale_factor: float           # longest-side / 256


def preprocess_image(
    image: Image.Image | np.ndarray,
    target_size: int = MEDSAM_INPUT_SIZE,
) -> PreprocessResult:
    """Prepare a single image for LiteMedSAM inference.

    Parameters
    ----------
    image : PIL Image or numpy HxWx3 uint8
    target_size : spatial side length (always 256 for the pretrained model)

    Returns
    -------
    PreprocessResult with the normalised, padded tensor and geometry metadata.
    """
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:  # greyscale → 3ch
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[2] == 1:
            arr = np.concatenate([arr] * 3, axis=-1)

    original_h, original_w = arr.shape[:2]

    # Resize longest side to target_size
    scale = target_size / max(original_h, original_w)
    new_h = int(round(original_h * scale))
    new_w = int(round(original_w * scale))
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Normalise to [0, 1]
    resized = resized.astype(np.float32)
    lo, hi = resized.min(), resized.max()
    if hi - lo > 1e-8:
        resized = (resized - lo) / (hi - lo)
    else:
        resized = np.zeros_like(resized)

    # Pad to (target_size, target_size, 3)
    padded = np.zeros((target_size, target_size, 3), dtype=np.float32)
    padded[:new_h, :new_w, :] = resized

    # (H, W, 3) → (1, 3, H, W)
    tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0)

    return PreprocessResult(
        tensor=tensor,
        original_hw=(original_h, original_w),
        resized_hw=(new_h, new_w),
        scale_factor=scale,
    )


def resize_box_to_medsam(
    box_xyxy: tuple[float, float, float, float],
    original_hw: tuple[int, int],
    target_size: int = MEDSAM_INPUT_SIZE,
) -> torch.Tensor:
    """Scale a bounding box from original image coordinates to 256×256.

    Returns a ``(1, 1, 4)`` tensor ready for ``MedSAM_Lite.forward(boxes=...)``.
    """
    oh, ow = original_hw
    scale = target_size / max(oh, ow)
    x1, y1, x2, y2 = box_xyxy
    scaled = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
    return torch.tensor(scaled, dtype=torch.float32).reshape(1, 1, 4)


# ── Heatmap → prompt geometry ─────────────────────────────────────────────

# Cut the tail of the activation distribution rather than using a fixed value.
# A fixed threshold is wrong in both directions: on a low-contrast scan nothing
# clears it and no box is produced, while on a high-contrast scan half the brain
# clears it and the box swallows the frame.
DEFAULT_CAM_PERCENTILE: float = 85.0

# Margin as a fraction of the box's own width/height. SAM tolerates a box that
# is slightly too loose; a box that clips the lesion edge costs far more Dice.
DEFAULT_BOX_MARGIN_FRAC: float = 0.10

# Elliptical opening kernel — removes speckle and thin satellite activations
# (skull, orbits) without eroding a compact lesion.
DEFAULT_MORPH_KERNEL: int = 5

# SAM needs a non-degenerate box. A component of one or two pixels would
# otherwise survive with a margin that rounds to zero.
DEFAULT_MIN_BOX_PX: int = 8

# A map where this fraction of pixels clears the cut is not a localised
# activation — it is a flat or saturated map, and any box drawn from it would be
# meaningless.
_MAX_BINARY_COVERAGE: float = 0.90


def _largest_component_box(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding rect of the largest connected component, or None if empty."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (DEFAULT_MORPH_KERNEL, DEFAULT_MORPH_KERNEL)
    )
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0:
        return None
    return cv2.boundingRect(largest)


def heatmap_to_box_prompt(
    cam: np.ndarray,
    resized_hw: tuple[int, int],
    threshold: float = 0.5,
    margin_px: int = 5,
    *,
    adaptive: bool = False,
    percentile: float = DEFAULT_CAM_PERCENTILE,
    margin_frac: float = DEFAULT_BOX_MARGIN_FRAC,
    min_box_px: int = DEFAULT_MIN_BOX_PX,
) -> torch.Tensor | None:
    """Extract a bounding box from a Grad-CAM heatmap in 256×256 space.

    Returns ``(1, 1, 4)`` tensor, or ``None`` when the map carries no activation.

    Two paths:

    **default (``adaptive=False``)**
        Fixed activation threshold plus a fixed pixel margin. This is the
        behaviour the workspace was validated with, and it is the default
        because the adaptive path below produced visibly looser boxes that made
        LiteMedSAM segment well beyond the lesion.
    **adaptive (``adaptive=True``)**
        Percentile cut → morphological opening → largest connected component →
        fractional margin. Kept because it cannot fail the way a fixed threshold
        does on a low-contrast scan, but it is opt-in: measured over the specimen
        samples it gave a wash on the decoder's IoU head while producing ~3×
        larger masks.

    Parameters
    ----------
    cam
        2D activation map, normalised to roughly [0, 1].
    resized_hw
        Accepted for API compatibility. The box is expressed in the model's
        256×256 prompt space, so the original image geometry is not needed here.
    threshold, margin_px
        Fixed-path cut and pad.
    adaptive
        Use the percentile/morphology path instead.
    percentile, margin_frac, min_box_px
        Adaptive-path knobs; ignored unless ``adaptive`` is set.

    Returns
    -------
    ``(1, 1, 4)`` float tensor of ``[x1, y1, x2, y2]``, or ``None``.
    """
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap, got shape {cam.shape}")

    target = MEDSAM_INPUT_SIZE
    resized = cv2.resize(cam, (target, target), interpolation=cv2.INTER_LINEAR)

    if not adaptive:
        # Original fixed-threshold path, unchanged.
        binary = (resized >= threshold).astype(np.uint8)
        coords = np.argwhere(binary)
        if coords.size == 0:
            return None
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        x_min = max(0, x_min - margin_px)
        y_min = max(0, y_min - margin_px)
        x_max = min(target - 1, x_max + margin_px)
        y_max = min(target - 1, y_max + margin_px)
        return torch.tensor(
            [float(x_min), float(y_min), float(x_max), float(y_max)],
            dtype=torch.float32,
        ).reshape(1, 1, 4)

    # ── Adaptive path ────────────────────────────────────────────────
    # An activation-free map must yield no prompt at all. This guard has to come
    # before the thresholding: the 85th percentile of an all-zero map is 0, so
    # ">= cut" would select every pixel and hand SAM a box around the whole
    # frame — a silent, plausible-looking failure.
    if float(resized.max()) <= 0.0 or float(resized.max() - resized.min()) <= 1e-6:
        return None

    cut = float(np.percentile(resized, percentile))
    if cut > 0.0:
        binary = (resized >= cut).astype(np.uint8)
    else:
        # The percentile landed on the floor, meaning more than
        # (100 - percentile)% of the map is exactly zero. ">=" at the floor
        # would select the entire frame, so take strictly-above-floor instead.
        binary = (resized > 0.0).astype(np.uint8)

    if binary.mean() > _MAX_BINARY_COVERAGE:
        return None

    rect = _largest_component_box(binary)
    if rect is None:
        return None

    x, y, w, h = rect
    pad_x = int(round(w * margin_frac))
    pad_y = int(round(h * margin_frac))

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(target - 1, x + w + pad_x)
    y2 = min(target - 1, y + h + pad_y)

    if (x2 - x1) < min_box_px or (y2 - y1) < min_box_px:
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        half = 0.5 * max(min_box_px, DEFAULT_MIN_BOX_PX)
        x1 = max(0, int(round(cx - half)))
        y1 = max(0, int(round(cy - half)))
        x2 = min(target - 1, int(round(cx + half)))
        y2 = min(target - 1, int(round(cy + half)))

    return torch.tensor(
        [float(x1), float(y1), float(x2), float(y2)], dtype=torch.float32
    ).reshape(1, 1, 4)


def heatmap_to_point_prompt(cam: np.ndarray) -> torch.Tensor | None:
    """Return the peak-activation point as ``(1, 1, 2)`` xy in 256×256 space.

    Used as a positive point prompt alongside the box. The box tells SAM where
    the lesion ends; the point tells it where the lesion definitely is, which
    stops the decoder latching onto a nearby bright structure of similar
    intensity inside the box.

    Returns ``None`` when the map carries no activation.
    """
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap, got shape {cam.shape}")

    target = MEDSAM_INPUT_SIZE
    resized = cv2.resize(cam, (target, target), interpolation=cv2.INTER_LINEAR)

    if float(resized.max()) <= 0.0:
        return None

    row, col = np.unravel_index(int(np.argmax(resized)), resized.shape)
    return torch.tensor(
        [float(col), float(row)], dtype=torch.float32
    ).reshape(1, 1, 2)


def heatmap_to_point_label() -> torch.Tensor:
    """Positive-point label tensor ``(1, 1)`` matching :func:`heatmap_to_point_prompt`."""
    return torch.ones((1, 1), dtype=torch.int64)


def postprocess_mask(
    low_res_logits: torch.Tensor,
    original_hw: tuple[int, int],
    resized_hw: tuple[int, int],
    target_size: int = MEDSAM_INPUT_SIZE,
) -> np.ndarray:
    """Upscale a low-res mask prediction back to the original image size.

    Parameters
    ----------
    low_res_logits : (1, 1, H, W) from the mask decoder
    original_hw : original image (H, W)
    resized_hw : image (H, W) after resize, before padding

    Returns
    -------
    binary_mask : (original_H, original_W) uint8 array, values in {0, 255}
    """
    # Upscale to 256×256
    mask_256 = F.interpolate(
        low_res_logits.float(),
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    # Crop to the pre-padding region
    rh, rw = resized_hw
    mask_cropped = mask_256[:, :, :rh, :rw]

    # Upscale to original dimensions
    oh, ow = original_hw
    mask_orig = F.interpolate(
        mask_cropped, size=(oh, ow), mode="bilinear", align_corners=False
    )
    # Binarise at logit > 0
    binary = (mask_orig.squeeze().cpu().numpy() > 0).astype(np.uint8) * 255
    return binary
