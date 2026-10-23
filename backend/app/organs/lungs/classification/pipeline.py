"""Serving pipeline for the Z-Net lung classifier.

Loads `checkpoints/lung_best_model.pth`, runs the paper's preprocessing
(resize 256 -> center crop 224 -> ImageNet normalisation), returns calibrated
probabilities for the 14 NIH findings plus a Grad-CAM overlay.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from .model import (
    FINDING_INDEX,
    FINDINGS,
    GradCAM,
    MODEL_TAG,
    ZNet,
    cam_to_bbox,
    overlay_cam_on_image,
)

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LungClassificationPipeline:
    """Lazily-loaded inference wrapper; the router owns the singleton."""

    def __init__(self, checkpoint_path: str, device: Optional[str] = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: Optional[ZNet] = None
        self.load_time_ms: Optional[float] = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    def load(self) -> None:
        if self.model is not None:
            return
        started = time.perf_counter()
        model = ZNet(len(FINDINGS), pretrained=False)
        state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning(
                "Checkpoint mismatch — missing=%s unexpected=%s", missing[:4], unexpected[:4]
            )
        model.eval().to(self.device)
        self.model = model
        self.load_time_ms = (time.perf_counter() - started) * 1000.0
        logger.info("Z-Net loaded in %.0f ms on %s", self.load_time_ms, self.device)

    def unload(self) -> None:
        self.model = None
        self.load_time_ms = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # ── inference ──────────────────────────────────────────────────────────

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        width, height = image.size
        scale = 256.0 / min(width, height)
        resized = image.resize(
            (max(256, int(round(width * scale))), max(256, int(round(height * scale)))),
            Image.BILINEAR,
        )
        left = (resized.width - 224) // 2
        top = (resized.height - 224) // 2
        cropped = resized.crop((left, top, left + 224, top + 224))
        array = np.asarray(cropped, dtype=np.float32) / 255.0
        array = (array - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(
            IMAGENET_STD, dtype=np.float32
        )
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image: Image.Image) -> dict[str, Any]:
        if self.model is None:
            self.load()
        assert self.model is not None
        started = time.perf_counter()
        tensor = self._preprocess(image)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.sigmoid(logits)[0].float().cpu().numpy()

        ranked = np.argsort(-probabilities)
        top5 = [
            {
                "finding": FINDINGS[int(index)],
                "probability": float(probabilities[int(index)]),
                "rank": rank + 1,
            }
            for rank, index in enumerate(ranked[:5])
        ]
        top_index = int(ranked[0])

        # Grad-CAM for the leading finding — the same map the UI renders.
        with GradCAM(self.model) as cam_gen:
            cam = cam_gen.generate(tensor.clone().requires_grad_(True), top_index)
        rgb = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.uint8)
        overlay = overlay_cam_on_image(rgb, cam, alpha=0.42)
        buffer = io.BytesIO()
        Image.fromarray(overlay).save(buffer, format="PNG", optimize=True)
        gradcam_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        bbox = cam_to_bbox(cam, threshold=0.6)
        return {
            "top_finding": FINDINGS[top_index],
            "confidence": float(probabilities[top_index]),
            "probabilities": [float(p) for p in probabilities],
            "top5": top5,
            "gradcam_base64": gradcam_base64,
            "gradcam_grid": [224, 224],
            "bbox": list(bbox) if bbox else None,
            "inference_ms": (time.perf_counter() - started) * 1000.0,
            "model_tag": MODEL_TAG,
        }

    # ── metadata ───────────────────────────────────────────────────────────

    def checkpoint_info(self) -> dict[str, Any]:
        try:
            stat = os.stat(self.checkpoint_path)
        except OSError:
            return {"present": False, "path": self.checkpoint_path}
        return {
            "present": True,
            "path": self.checkpoint_path,
            "size_mb": round(stat.st_size / (1024 * 1024), 1),
            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
        }


__all__ = ["LungClassificationPipeline", "FINDINGS", "FINDING_INDEX"]
