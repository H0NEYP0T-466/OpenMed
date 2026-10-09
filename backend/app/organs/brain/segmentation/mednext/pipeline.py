"""MedNeXt serving pipeline: checkpoint loading and single-image inference.

The checkpoint is self-describing (see ``training/export.py``): architecture,
preprocessing contract and tuned post-processing all travel with the weights, so
dropping ``mednext_brain_seg.pth`` into ``checkpoints/`` is the whole
deployment. Loading is strict and fails closed with a precise reason.

The same class is used by training for validation-time tuning and for the final
held-out test, so the number reported is the number the application produces.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import cv2
import numpy as np
import torch

from .model import MedNeXtSpec, build_model, count_parameters
from .preprocessor import (
    PREPROCESS_VERSION,
    Geometry,
    canvas_to_original,
    decode_gray,
    normalise,
    to_canvas,
    to_gray_uint8,
)

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "mednext_brain_seg.pth"
CHECKPOINT_ENV = "OPENMED_MEDNEXT_CHECKPOINT"
SUPPORTED_FORMAT = 1
OUTPUT_MAX_SIDE = 768
OVERLAY_COLOUR = (237, 111, 92)


class SegmentationUnavailableError(RuntimeError):
    """The MedNeXt checkpoint is missing or unusable. Serving fails closed."""


def default_checkpoint_path() -> str:
    configured = os.getenv(CHECKPOINT_ENV)
    if configured:
        return os.path.expanduser(configured)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", CHECKPOINT_NAME)


def load_serving_checkpoint(path: str) -> dict[str, Any]:
    """Read and validate a serving checkpoint without building the network."""
    if not os.path.isfile(path):
        raise SegmentationUnavailableError(
            f"MedNeXt checkpoint not found at {path}. Train it with train_mednext.py on Kaggle "
            f"and place {CHECKPOINT_NAME} in that folder."
        )
    if os.path.getsize(path) == 0:
        raise SegmentationUnavailableError(f"MedNeXt checkpoint at {path} is empty.")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - any unreadable file is fail-closed
        raise SegmentationUnavailableError(f"MedNeXt checkpoint is unreadable: {exc}") from exc

    if not isinstance(payload, dict) or payload.get("kind") != "mednext_2d_brain_binary_seg":
        raise SegmentationUnavailableError(
            "File is not a MedNeXt serving checkpoint (use best/mednext_brain_seg.pth, not last.pt)."
        )
    if payload.get("format_version") != SUPPORTED_FORMAT:
        raise SegmentationUnavailableError(
            f"Unsupported checkpoint format {payload.get('format_version')!r}; "
            f"this build reads format {SUPPORTED_FORMAT}."
        )
    version = payload.get("preprocess", {}).get("version")
    if version != PREPROCESS_VERSION:
        raise SegmentationUnavailableError(
            f"Checkpoint was trained with preprocessing v{version}; this build implements "
            f"v{PREPROCESS_VERSION}. Serving it would feed the network inputs it never saw."
        )
    return payload


@dataclass
class SegmentationResult:
    mask: np.ndarray
    probability: np.ndarray
    original_size: tuple[int, int]
    input_size: int
    threshold: float
    min_area_frac: float
    tta: bool
    inference_ms: float
    foreground_px: int = 0
    foreground_fraction: float = 0.0
    mean_probability: float = 0.0
    peak_probability: float = 0.0
    n_components: int = 0
    bbox: Optional[list[int]] = None
    extra: dict[str, Any] = field(default_factory=dict)


def postprocess_mask(
    probability: np.ndarray, threshold: float, min_area_frac: float
) -> np.ndarray:
    """Threshold, then drop specks below ``min_area_frac`` of the frame.

    If filtering would remove every component, the largest one is kept: the
    network found *something*, and returning an empty mask for a scan it flagged
    would hide that.
    """
    mask = (probability >= threshold).astype(np.uint8)
    if min_area_frac <= 0.0 or not mask.any():
        return mask
    minimum = max(1, int(round(min_area_frac * mask.size)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, area in enumerate(areas) if area >= minimum]
    if not keep:
        keep = [1 + int(np.argmax(areas))]
    return np.isin(labels, keep).astype(np.uint8)


def _downscale(image: np.ndarray, max_side: int) -> np.ndarray:
    longest = max(image.shape[:2])
    if longest <= max_side:
        return image
    scale = max_side / longest
    size = (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def encode_mask_png(mask: np.ndarray, max_side: int = OUTPUT_MAX_SIDE) -> str:
    plane = _downscale((mask > 0).astype(np.uint8) * 255, max_side)
    plane = (plane > 127).astype(np.uint8) * 255
    ok, buffer = cv2.imencode(".png", plane)
    if not ok:
        raise RuntimeError("PNG encoding failed for the segmentation mask.")
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def encode_overlay(
    gray: np.ndarray, mask: np.ndarray, max_side: int = OUTPUT_MAX_SIDE, alpha: float = 0.45
) -> str:
    base = _downscale(gray, max_side)
    small = cv2.resize(mask.astype(np.uint8), (base.shape[1], base.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB).astype(np.float32)
    colour = np.array(OVERLAY_COLOUR, dtype=np.float32)
    rgb[small] = (1.0 - alpha) * rgb[small] + alpha * colour
    out = np.clip(rgb, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(small.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, OVERLAY_COLOUR, thickness=2)
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("JPEG encoding failed for the segmentation overlay.")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


class MedNeXtSegmenter:
    """Loads a serving checkpoint and segments one image at a time."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None) -> None:
        self.model_path = model_path or default_checkpoint_path()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[torch.nn.Module] = None
        self._load()

    def _load(self) -> None:
        payload = load_serving_checkpoint(self.model_path)
        try:
            self.spec = MedNeXtSpec.from_dict(payload["spec"])
            model = build_model(self.spec, deep_supervision=False)
            model.load_state_dict(payload["state_dict"], strict=True)
        except Exception as exc:  # noqa: BLE001 - mismatched weights must fail closed
            raise SegmentationUnavailableError(
                f"MedNeXt weights do not match the architecture they declare: {exc}"
            ) from exc

        self.model = model.to(self.device).eval()
        self.size = int(payload["preprocess"]["image_size"])
        self.postprocess = dict(payload["postprocess"])
        self.metrics = payload.get("metrics", {})
        self.training = payload.get("training", {})
        self.parameters = count_parameters(model)
        logger.info(
            "MedNeXt-%s loaded: %.2fM params, input %dpx, threshold %.2f, min_area %.4f, tta %s, device %s",
            self.spec.variant, self.parameters / 1e6, self.size,
            self.postprocess.get("threshold", 0.5), self.postprocess.get("min_area_frac", 0.0),
            self.postprocess.get("tta_hflip", False), self.device,
        )

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def unload(self) -> None:
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _forward(self, canvases: list[np.ndarray]) -> np.ndarray:
        if self.model is None:
            raise SegmentationUnavailableError("MedNeXt model is not loaded.")
        batch = torch.from_numpy(np.stack([normalise(c) for c in canvases])[:, None]).to(self.device)
        use_amp = self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=use_amp):
            logits = self.model(batch)
        return torch.sigmoid(logits.float())[:, 0].cpu().numpy()

    def canvas_probabilities(
        self, canvas: np.ndarray, geom: Geometry, tta: bool
    ) -> np.ndarray:
        """Probabilities on the model canvas. Flip-TTA mirrors only the valid region."""
        variants = [canvas]
        if tta:
            flipped = np.zeros_like(canvas)
            flipped[: geom.scaled_h, : geom.scaled_w] = canvas[: geom.scaled_h, : geom.scaled_w][:, ::-1]
            variants.append(flipped)
        probs = self._forward(variants)
        merged = probs[0].copy()
        if tta:
            back = np.zeros_like(merged)
            back[: geom.scaled_h, : geom.scaled_w] = probs[1][: geom.scaled_h, : geom.scaled_w][:, ::-1]
            merged = 0.5 * (merged + back)
        return merged

    def predict_gray(
        self,
        gray: np.ndarray,
        *,
        threshold: Optional[float] = None,
        min_area_frac: Optional[float] = None,
        tta: Optional[bool] = None,
    ) -> SegmentationResult:
        started = time.perf_counter()
        gray = to_gray_uint8(gray)
        threshold = float(self.postprocess.get("threshold", 0.5) if threshold is None else threshold)
        min_area = float(
            self.postprocess.get("min_area_frac", 0.0) if min_area_frac is None else min_area_frac
        )
        use_tta = bool(self.postprocess.get("tta_hflip", False) if tta is None else tta)

        canvas, geom = to_canvas(gray, self.size)
        canvas_probs = self.canvas_probabilities(canvas, geom, use_tta)
        probability = canvas_to_original(canvas_probs, geom).astype(np.float32)
        mask = postprocess_mask(probability, threshold, min_area)

        result = SegmentationResult(
            mask=mask, probability=probability,
            original_size=(geom.original_h, geom.original_w), input_size=self.size,
            threshold=threshold, min_area_frac=min_area, tta=use_tta,
            inference_ms=(time.perf_counter() - started) * 1000.0,
        )
        foreground = int(mask.sum())
        result.foreground_px = foreground
        result.foreground_fraction = foreground / mask.size
        result.peak_probability = float(probability.max())
        if foreground:
            result.mean_probability = float(probability[mask > 0].mean())
            result.n_components = int(cv2.connectedComponents(mask, connectivity=8)[0] - 1)
            ys, xs = np.nonzero(mask)
            result.bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        return result

    def predict_bytes(self, data: bytes, **kwargs: Any) -> tuple[SegmentationResult, np.ndarray]:
        """Decode with the training-time decoder, then segment. Returns (result, gray)."""
        gray = decode_gray(data)
        return self.predict_gray(gray, **kwargs), gray

    def predict(self, image: Union[bytes, np.ndarray, Any], **kwargs: Any) -> SegmentationResult:
        if isinstance(image, (bytes, bytearray)):
            return self.predict_bytes(bytes(image), **kwargs)[0]
        return self.predict_gray(to_gray_uint8(image), **kwargs)

    def model_info(self) -> dict[str, Any]:
        return {
            "model_name": f"MedNeXt-{self.spec.variant} (2D)",
            "architecture": self.spec.to_dict(),
            "parameters": self.parameters,
            "input_size": f"{self.size}x{self.size}",
            "channels": 1,
            "postprocess": self.postprocess,
            "metrics": self.metrics,
            "training": self.training,
            "device": str(self.device),
            "checkpoint": os.path.basename(self.model_path),
        }
