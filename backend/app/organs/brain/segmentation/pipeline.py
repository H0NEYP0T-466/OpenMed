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
    heatmap_to_point_label,
    heatmap_to_point_prompt,
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


# ── Click-to-segment prompt geometry ─────────────────────────────────────
#
# Side of the box built around a doctor's click, in 256² prompt space. 48 was
# the best of {48, 64, 96} measured against ground truth (median Dice 0.888 vs
# 0.698 at 64 and 0.370 at 96) — larger boxes fall off the same cliff as the
# loose adaptive box did.
DEFAULT_CLICK_BOX_PX = 48
MIN_CLICK_BOX_PX = 16
MAX_CLICK_BOX_PX = 192


class BrainSegmentationPipeline:
    """End-to-end inference for brain tumour segmentation with LiteMedSAM.

    Prompt modes:
    - **heatmap_box** (``segment_with_heatmap``): a bounding box extracted from
      the Grad-CAM heatmap is used as the prompt. The released LiteMedSAM
      weights segment from boxes, which is why this is the default.
    - **raw** (``segment_raw``): no prompt at all. The decoder runs from the
      image embedding alone. Offered as the unprompted arm of the ablation.
    - **box** (``segment_with_box``): an explicit caller-supplied box.
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
        use_peak_point: bool = False,
        adaptive_box: bool = False,
    ) -> dict[str, Any]:
        """Segment a tumour using the Grad-CAM heatmap as a prompt.

        The released LiteMedSAM weights prompt via a bounding box only (they
        were never trained on dense masks), so the heatmap is reduced to prompt
        geometry and run once. The segmentation result is returned downscaled to
        keep the response payload small for the browser.

        The box uses the fixed activation threshold plus a fixed pixel margin by
        default — that is the configuration the workspace was validated with. An
        adaptive percentile/morphology path exists behind ``adaptive_box``, but
        it is off by default: it produced visibly looser boxes that made the
        decoder segment well beyond the lesion.

        Parameters
        ----------
        image : path or PIL Image (the original uploaded image)
        cam_heatmap : 2D normalised [0, 1] array from the classifier's CAM
        cam_threshold : activation cut for the fixed box path
        box_margin_px : pixel pad for the fixed box path
        use_peak_point : also prompt with the peak-activation point. Off by
            default — measured 0.051 worse mean IoU when sent unconditionally.
        adaptive_box : use the percentile/morphology box path instead

        Returns
        -------
        dict with keys: box_mask_base64, box_overlay_base64, iou_box,
                        box_coords, point_prompt_used, point_coords,
                        mask_foreground_px, input_size, original_size.
        """
        if not self._loaded or self.model is None:
            raise SegmentationUnavailableError("Segmentation model is not loaded.")

        pil_image = self._load_image(image)
        prep = preprocess_image(pil_image, MEDSAM_INPUT_SIZE)
        input_tensor = prep.tensor.to(self.device)

        result: dict[str, Any] = {
            "segmentation_performed": True,
            "prompt_mode": "heatmap_box",
            "input_size": f"{MEDSAM_INPUT_SIZE}x{MEDSAM_INPUT_SIZE}",
            "original_size": f"{prep.original_hw[1]}x{prep.original_hw[0]}",
        }

        # ── Box prompt segmentation ───────────────────────────────────
        box_tensor = heatmap_to_box_prompt(
            cam_heatmap,
            resized_hw=prep.resized_hw,
            threshold=cam_threshold,
            margin_px=box_margin_px,
            adaptive=adaptive_box,
        )

        if box_tensor is not None:
            box_tensor = box_tensor.to(self.device)

            point_prompt: Optional[tuple[torch.Tensor, torch.Tensor]] = None
            if use_peak_point:
                point_coords = heatmap_to_point_prompt(cam_heatmap)
                if point_coords is not None:
                    point_prompt = (
                        point_coords.to(self.device),
                        heatmap_to_point_label().to(self.device),
                    )

            with torch.no_grad():
                low_res_masks, iou_pred = self.model(
                    input_tensor, boxes=box_tensor, points=point_prompt,
                )
            mask_box = postprocess_mask(low_res_masks, prep.original_hw, prep.resized_hw)
            result["box_mask_base64"] = _encode_mask_png(mask_box, max_side=SEG_MAX_SIDE)
            result["box_overlay_base64"] = _encode_overlay(
                pil_image, mask_box, max_side=SEG_MAX_SIDE,
            )
            result["iou_box"] = round(float(iou_pred.squeeze().cpu().item()), 4)
            result["box_coords"] = box_tensor.squeeze().cpu().tolist()
            result["mask_foreground_px"] = _foreground_px(mask_box)
            result["box_prompt_used"] = True
            result["point_prompt_used"] = point_prompt is not None
            if point_prompt is not None:
                result["point_coords"] = point_prompt[0].squeeze().cpu().tolist()
        else:
            result["box_prompt_used"] = False
            result["point_prompt_used"] = False
            result["box_prompt_note"] = (
                "No region in the heatmap exceeded the activation threshold."
            )

        return result

    def segment_raw(
        self,
        image: Union[str, Image.Image],
    ) -> dict[str, Any]:
        """Run LiteMedSAM with no prompt at all.

        The prompt encoder receives no points, no box and no dense mask, so the
        mask decoder works from the image embedding alone (its ``no_mask_embed``
        dense embedding plus an empty sparse token set). This is the unprompted
        arm of the heatmap-prompt ablation.

        The released ``lite_medsam.pth`` weights were trained to segment from
        bounding boxes, so an unprompted forward is expected to score worse than
        the heatmap-prompted path. That is the point of offering it: the caller
        can show the difference rather than assert it.

        Returns
        -------
        dict with keys: seg_mask_base64, seg_overlay_base64, iou_pred,
                        prompt_mode, input_size, original_size.
        """
        if not self._loaded or self.model is None:
            raise SegmentationUnavailableError("Segmentation model is not loaded.")

        pil_image = self._load_image(image)
        prep = preprocess_image(pil_image, MEDSAM_INPUT_SIZE)
        input_tensor = prep.tensor.to(self.device)

        with torch.no_grad():
            low_res_masks, iou_pred = self.model(input_tensor)

        mask = postprocess_mask(low_res_masks, prep.original_hw, prep.resized_hw)

        return {
            "segmentation_performed": True,
            "prompt_mode": "raw",
            "seg_mask_base64": _encode_mask_png(mask, max_side=SEG_MAX_SIDE),
            "seg_overlay_base64": _encode_overlay(
                pil_image, mask, max_side=SEG_MAX_SIDE,
            ),
            "iou_pred": round(float(iou_pred.squeeze().cpu().item()), 4),
            "mask_foreground_px": _foreground_px(mask),
            "input_size": f"{MEDSAM_INPUT_SIZE}x{MEDSAM_INPUT_SIZE}",
            "original_size": f"{prep.original_hw[1]}x{prep.original_hw[0]}",
        }

    def segment_with_click(
        self,
        image: Union[str, Image.Image],
        click_x: float,
        click_y: float,
        box_size: int = DEFAULT_CLICK_BOX_PX,
    ) -> dict[str, Any]:
        """Segment from a single click on the lesion.

        The doctor marks the suspicious region; that (x, y) becomes a small box
        centred on it and the model segments inside it. This is the assistive
        path: the clinician supplies the localisation the CAM cannot.

        The click is converted to a **box**, not sent as a bare point. Measured
        against ground truth on the held-out split: a 48px box centred on the
        click reached a median Dice of 0.888, while the same click sent as a bare
        point scored 0.254 — the released weights were trained on boxes only and
        respond weakly to points. Sending the point *alongside* the box also
        measured worse in every case, so the box is sent alone.

        Parameters
        ----------
        image : path or PIL Image (the original uploaded image)
        click_x, click_y : click position in 256×256 prompt space
        box_size : side of the square box built around the click

        Returns
        -------
        dict with keys: seg_mask_base64, seg_overlay_base64, iou_pred,
                        mask_foreground_px, click, box_coords, prompt_mode,
                        input_size, original_size.
        """
        if not self._loaded or self.model is None:
            raise SegmentationUnavailableError("Segmentation model is not loaded.")
        if not (MIN_CLICK_BOX_PX <= box_size <= MAX_CLICK_BOX_PX):
            raise ValueError(
                f"box_size must be between {MIN_CLICK_BOX_PX} and "
                f"{MAX_CLICK_BOX_PX}, got {box_size}."
            )

        half = box_size / 2.0
        # Centre the box on the click, but never let clamping shrink it: a box
        # smaller than requested clips the lesion, and the box-quality curve
        # showed smaller boxes cost Dice. Near an edge the box shifts instead,
        # so it stays full-size and the click stays inside it.
        limit = 255.0 - box_size
        x1 = min(max(click_x - half, 0.0), limit)
        y1 = min(max(click_y - half, 0.0), limit)
        x2 = x1 + box_size
        y2 = y1 + box_size
        box = torch.tensor([x1, y1, x2, y2], dtype=torch.float32).reshape(1, 1, 4).to(self.device)

        pil_image = self._load_image(image)
        prep = preprocess_image(pil_image, MEDSAM_INPUT_SIZE)

        with torch.no_grad():
            low_res_masks, iou_pred = self.model(prep.tensor.to(self.device), boxes=box)

        mask = postprocess_mask(low_res_masks, prep.original_hw, prep.resized_hw)

        return {
            "segmentation_performed": True,
            "prompt_mode": "click_box",
            "seg_mask_base64": _encode_mask_png(mask, max_side=SEG_MAX_SIDE),
            "seg_overlay_base64": _encode_overlay(
                pil_image, mask, max_side=SEG_MAX_SIDE,
            ),
            "iou_pred": round(float(iou_pred.squeeze().cpu().item()), 4),
            "mask_foreground_px": _foreground_px(mask),
            "click": [float(click_x), float(click_y)],
            "box_coords": box.squeeze().cpu().tolist(),
            "input_size": f"{MEDSAM_INPUT_SIZE}x{MEDSAM_INPUT_SIZE}",
            "original_size": f"{prep.original_hw[1]}x{prep.original_hw[0]}",
        }

    def segment_with_drawn_box(
        self,
        image: Union[str, Image.Image],
        x1n: float,
        y1n: float,
        x2n: float,
        y2n: float,
    ) -> dict[str, Any]:
        """Segment inside a box the clinician drew themselves.

        The assistive counterpart to :meth:`segment_with_click`: instead of
        deriving a box from a single click, the clinician drags one around the
        lesion. Coordinates are normalised 0-1 against the displayed scan and
        converted to 256×256 prompt space, the same convention as the click
        path, so both interactions are interchangeable from the UI's point of
        view.

        This is the prompt type the released weights were trained on, so a
        well-drawn box is the strongest prompt available — measured at 0.89
        Dice against ground truth for a box roughly matching the lesion.

        Returns
        -------
        dict with keys: seg_mask_base64, seg_overlay_base64, iou_pred,
                        mask_foreground_px, box_coords, prompt_mode,
                        input_size, original_size.
        """
        if not self._loaded or self.model is None:
            raise SegmentationUnavailableError("Segmentation model is not loaded.")

        # Normalised -> 256² prompt space, then ordered and clamped. A drag
        # that starts bottom-right and ends top-left is still a valid box.
        x1, x2 = sorted((float(x1n) * 255.0, float(x2n) * 255.0))
        y1, y2 = sorted((float(y1n) * 255.0, float(y2n) * 255.0))

        # A box smaller than this cannot enclose a lesion meaningfully, and the
        # box-quality curve showed tiny boxes score badly.
        if (x2 - x1) < MIN_CLICK_BOX_PX or (y2 - y1) < MIN_CLICK_BOX_PX:
            raise ValueError(
                f"Drawn box is too small — draw a larger region (minimum "
                f"{MIN_CLICK_BOX_PX}px in 256² space)."
            )

        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(255.0, x2), min(255.0, y2)
        box = torch.tensor([x1, y1, x2, y2], dtype=torch.float32).reshape(1, 1, 4).to(self.device)

        pil_image = self._load_image(image)
        prep = preprocess_image(pil_image, MEDSAM_INPUT_SIZE)

        with torch.no_grad():
            low_res_masks, iou_pred = self.model(prep.tensor.to(self.device), boxes=box)

        mask = postprocess_mask(low_res_masks, prep.original_hw, prep.resized_hw)

        return {
            "segmentation_performed": True,
            "prompt_mode": "drawn_box",
            "seg_mask_base64": _encode_mask_png(mask, max_side=SEG_MAX_SIDE),
            "seg_overlay_base64": _encode_overlay(
                pil_image, mask, max_side=SEG_MAX_SIDE,
            ),
            "iou_pred": round(float(iou_pred.squeeze().cpu().item()), 4),
            "mask_foreground_px": _foreground_px(mask),
            "box_coords": box.squeeze().cpu().tolist(),
            "input_size": f"{MEDSAM_INPUT_SIZE}x{MEDSAM_INPUT_SIZE}",
            "original_size": f"{prep.original_hw[1]}x{prep.original_hw[0]}",
        }

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
            "mask_foreground_px": _foreground_px(mask),
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
            # What the serving pipeline can actually be asked to do. "raw" sends
            # no prompt at all; it is the unprompted arm of the ablation, not a
            # prompt type the weights were trained on.
            "serving_prompt_modes": ["heatmap_box", "raw"],
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


def _foreground_px(mask: np.ndarray) -> int:
    """Count foreground pixels in a binary ``{0, 255}`` mask.

    Reported alongside every mask so a caller can tell an *empty* prediction
    apart from a missing one. Without it a zero-pixel mask arrives as a
    perfectly valid, entirely black PNG and looks like a rendering fault rather
    than the model finding nothing — which is exactly what an unprompted
    LiteMedSAM forward tends to produce.
    """
    return int(np.count_nonzero(mask > 127))


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
