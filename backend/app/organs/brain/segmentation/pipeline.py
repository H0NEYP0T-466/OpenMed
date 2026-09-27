"""Brain tumour segmentation pipeline.

Wraps LiteMedSAM into a self-contained inference pipeline that:

1. Loads the ``lite_medsam.pth`` checkpoint on first use.
2. Accepts a PIL image and optional Grad-CAM heatmap.
3. Converts the heatmap into a bounding-box prompt and runs segmentation
   with the box (the released LiteMedSAM weights prompt via boxes only).
4. Produces a binary mask, an overlay image, and base64-encoded outputs
   ready for the API response.

The pipeline is designed for lazy loading and explicit unloading so that
memory is freed when the user navigates away from the brain section.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from .model import MedSAM_Lite, build_medsam_lite
from .preprocessor import (
    MEDSAM_INPUT_SIZE,
    heatmap_to_box_prompt,
    postprocess_mask,
    preprocess_image,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BrainSegmentationPipeline",
    "SegmentationUnavailableError",
]


class SegmentationUnavailableError(RuntimeError):
    """The segmentation model could not be loaded or is missing."""


class BrainSegmentationPipeline:
    """End-to-end inference for brain tumour segmentation with LiteMedSAM.

    Prompt mode:
    - **box prompt**: a bounding box extracted from the Grad-CAM heatmap.
      The released LiteMedSAM weights segment from boxes, which is why this
      is the only prompt mode the pipeline runs.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model: Optional[MedSAM_Lite] = None
        self._loaded = False

        if model_path:
            self._load(model_path)

    def _load(self, model_path: str) -> None:
        """Build the model and load the checkpoint."""
        if not os.path.isfile(model_path):
            raise SegmentationUnavailableError(
                f"Segmentation checkpoint not found at {model_path}. "
                "Run: python -m app.organs.brain.segmentation.prepare_weights"
            )

        logger.info("Building LiteMedSAM model architecture...")
        self.model = build_medsam_lite()

        logger.info("Loading segmentation weights from %s", model_path)
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise SegmentationUnavailableError(
                f"Could not read checkpoint {model_path}: {exc}"
            ) from exc

        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise SegmentationUnavailableError(
                f"Checkpoint is incompatible with the LiteMedSAM architecture: {exc}"
            ) from exc

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True

        logger.info(
            "Segmentation pipeline ready  device=%s  input=%dx%d",
            self.device, MEDSAM_INPUT_SIZE, MEDSAM_INPUT_SIZE,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        """Release model weights and free GPU/CPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Segmentation model unloaded, memory freed.")

    def segment_with_heatmap(
        self,
        image: Union[str, Image.Image],
        cam_heatmap: np.ndarray,
        cam_threshold: float = 0.5,
        box_margin_px: int = 10,
    ) -> dict[str, Any]:
        """Segment a tumour using the Grad-CAM heatmap as a prompt.

        The released LiteMedSAM weights prompt via a bounding box only (they
        were never trained on dense masks), so the heatmap is converted into a
        single box prompt and run once.  The segmentation result is returned
        downscaled to keep the response payload small for the browser.

        Parameters
        ----------
        image : path or PIL Image (the original uploaded image)
        cam_heatmap : 2D normalised [0, 1] array from the classifier's Grad-CAM
        cam_threshold : threshold for deriving the box from the heatmap
        box_margin_px : padding around the box prompt

        Returns
        -------
        dict with keys: box_mask_base64, box_overlay_base64, iou_box,
                        box_coords, input_size, original_size.
        """
        if not self._loaded or self.model is None:
            raise SegmentationUnavailableError("Segmentation model is not loaded.")

        pil_image = self._load_image(image)
        prep = preprocess_image(pil_image, MEDSAM_INPUT_SIZE)
        input_tensor = prep.tensor.to(self.device)

        result: dict[str, Any] = {
            "segmentation_performed": True,
            "input_size": f"{MEDSAM_INPUT_SIZE}x{MEDSAM_INPUT_SIZE}",
            "original_size": f"{prep.original_hw[1]}x{prep.original_hw[0]}",
        }

        # ── Box prompt segmentation ───────────────────────────────────
        box_tensor = heatmap_to_box_prompt(
            cam_heatmap,
            resized_hw=prep.resized_hw,
            threshold=cam_threshold,
            margin_px=box_margin_px,
        )

        if box_tensor is not None:
            box_tensor = box_tensor.to(self.device)
            with torch.no_grad():
                low_res_masks, iou_pred = self.model(
                    input_tensor, boxes=box_tensor,
                )
            mask_box = postprocess_mask(low_res_masks, prep.original_hw, prep.resized_hw)
            result["box_mask_base64"] = _encode_mask_png(mask_box, max_side=SEG_MAX_SIDE)
            result["box_overlay_base64"] = _encode_overlay(
                pil_image, mask_box, max_side=SEG_MAX_SIDE,
            )
            result["iou_box"] = round(float(iou_pred.squeeze().cpu().item()), 4)
            result["box_coords"] = box_tensor.squeeze().cpu().tolist()
            result["box_prompt_used"] = True
        else:
            result["box_prompt_used"] = False
            result["box_prompt_note"] = (
                "No region in the heatmap exceeded the activation threshold."
            )

        return result

    def segment_with_box(
        self,
        image: Union[str, Image.Image],
        box_xyxy: tuple[float, float, float, float],
    ) -> dict[str, Any]:
        """Run segmentation with an explicit bounding box in original image coords.

        This is the fallback for when no heatmap is available.
        """
        if not self._loaded or self.model is None:
            raise SegmentationUnavailableError("Segmentation model is not loaded.")

        pil_image = self._load_image(image)
        prep = preprocess_image(pil_image, MEDSAM_INPUT_SIZE)
        input_tensor = prep.tensor.to(self.device)

        # Scale box to 256×256 space
        scale = prep.scale_factor
        scaled_box = [
            box_xyxy[0] * scale,
            box_xyxy[1] * scale,
            box_xyxy[2] * scale,
            box_xyxy[3] * scale,
        ]
        box_tensor = torch.tensor(scaled_box, dtype=torch.float32).reshape(1, 1, 4).to(self.device)

        with torch.no_grad():
            low_res_masks, iou_pred = self.model(input_tensor, boxes=box_tensor)

        mask = postprocess_mask(low_res_masks, prep.original_hw, prep.resized_hw)

        return {
            "segmentation_performed": True,
            "box_mask_base64": _encode_mask_png(mask, max_side=SEG_MAX_SIDE),
            "box_overlay_base64": _encode_overlay(pil_image, mask, max_side=SEG_MAX_SIDE),
            "iou_box": round(float(iou_pred.squeeze().cpu().item()), 4),
            "box_coords": scaled_box,
            "input_size": f"{MEDSAM_INPUT_SIZE}x{MEDSAM_INPUT_SIZE}",
            "original_size": f"{prep.original_hw[1]}x{prep.original_hw[0]}",
        }

    @staticmethod
    def _load_image(image_source: Union[str, Image.Image]) -> Image.Image:
        if isinstance(image_source, str):
            return Image.open(image_source).convert("RGB")
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
        raise TypeError(f"Expected a path or PIL.Image, got {type(image_source).__name__}")

    def model_info(self) -> dict[str, Any]:
        return {
            "model_name": "LiteMedSAM",
            "architecture": "TinyViT-256 + SAM PromptEncoder + MaskDecoder",
            "input_size": f"{MEDSAM_INPUT_SIZE}x{MEDSAM_INPUT_SIZE}",
            # The architecture accepts dense masks and points, but the released
            # lite_medsam.pth weights were only trained to segment from boxes.
            "prompt_types": ["bounding_box"],
            "loaded": self._loaded,
            "device": str(self.device) if self._loaded else None,
        }


# ── Encoding helpers ──────────────────────────────────────────────────────

# Rendered output is downscaled to this side before encoding.  The browser
# only ever shows these tiles at ~half page width, so a 512×512 PNG/JPEG was
# pure overhead (and the main cause of the laggy scroll from big base64 data
# URLs).  384 keeps the tumour crisp while cutting the payload ~2×; 256 is
# lighter still if you want max smoothness.
SEG_MAX_SIDE = 384


def _downscale(img: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink an image so its longest side is at most ``max_side``."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / longest
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    interp = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_AREA
    return cv2.resize(img, new_size, interpolation=interp)


def _encode_mask_png(mask: np.ndarray, max_side: int = SEG_MAX_SIDE) -> str:
    """Encode a binary mask as a base64 data-URL PNG, downscaled to fit."""
    mask = _downscale(mask, max_side)
    success, buffer = cv2.imencode(".png", mask)
    if not success:
        raise RuntimeError("PNG encoding failed for the segmentation mask.")
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def _encode_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (237, 111, 92),  # coral accent
    alpha: float = 0.45,
    quality: int = 85,
    max_side: int = SEG_MAX_SIDE,
) -> str:
    """Blend a coloured segmentation mask onto the original image.

    Uses the OpenMed coral accent colour by default.  The blended raster is
    downscaled to ``max_side`` and JPEG-compressed at ``quality`` so the
    returned data URL stays small.
    """
    base = np.array(image.convert("RGB"))
    if mask.shape[:2] != base.shape[:2]:
        mask = cv2.resize(mask, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = base.copy().astype(np.float32)
    colour = np.array(color, dtype=np.float32)
    mask_bool = mask > 127

    overlay[mask_bool] = (1.0 - alpha) * overlay[mask_bool] + alpha * colour
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Draw contour for crisp edge
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, thickness=2)

    overlay = _downscale(overlay, max_side)
    success, buffer = cv2.imencode(
        ".jpg",
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not success:
        raise RuntimeError("JPEG encoding failed for the segmentation overlay.")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")
