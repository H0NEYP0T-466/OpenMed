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


def heatmap_to_mask_prompt(
    cam: np.ndarray,
    target_size: int = MEDSAM_INPUT_SIZE,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Convert a Grad-CAM heatmap into a dense mask prompt for MedSAM.

    The heatmap is resized to 256×256, thresholded, and returned as a
    ``(1, 1, 256, 256)`` float tensor suitable for
    ``prompt_encoder(masks=...)``.

    Parameters
    ----------
    cam : 2D numpy array, normalised [0, 1] (from GradCAM.generate)
    target_size : spatial side (256)
    threshold : activation threshold — regions above this become the prompt

    Returns
    -------
    mask_prompt : (1, 1, 256, 256) float32 tensor
    """
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap, got shape {cam.shape}")

    resized = cv2.resize(cam, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    # Binarise at threshold, then keep soft values above it
    mask = np.where(resized >= threshold, resized, 0.0).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def heatmap_to_box_prompt(
    cam: np.ndarray,
    resized_hw: tuple[int, int],
    threshold: float = 0.5,
    margin_px: int = 5,
) -> torch.Tensor | None:
    """Extract a bounding box from a Grad-CAM heatmap in 256×256 space.

    Returns ``(1, 1, 4)`` tensor or ``None`` if no region passes threshold.
    """
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap, got shape {cam.shape}")

    target = MEDSAM_INPUT_SIZE
    resized = cv2.resize(cam, (target, target), interpolation=cv2.INTER_LINEAR)
    binary = (resized >= threshold).astype(np.uint8)

    coords = np.argwhere(binary)
    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Add margin, clamp
    x_min = max(0, x_min - margin_px)
    y_min = max(0, y_min - margin_px)
    x_max = min(target - 1, x_max + margin_px)
    y_max = min(target - 1, y_max + margin_px)

    return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32).reshape(1, 1, 4)


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
