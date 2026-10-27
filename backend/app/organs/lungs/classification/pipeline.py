"""Serving pipeline for the XRV 16-class lung classifier.

Loads `checkpoints/xrv_best.pth` — a TorchXRayVision DenseNet-121 fine-tuned on
NIH + COVID + TB — and returns raw sigmoid posteriors for the 16 classes plus a
Grad-CAM overlay.

Preprocessing must match training exactly: one grayscale channel, resized to
224, scaled to [-1024, 1024]. Any deviation here silently degrades every
prediction, so the constants live in `model.py` beside the architecture.
"""

from __future__ import annotations

import base64
import io
import json
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
    INPUT_SIZE,
    MODEL_TAG,
    GradCAM,
    build_model,
    cam_to_bbox,
    overlay_cam_on_image,
)

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5


def load_thresholds(checkpoint_path: str) -> Optional[list[float]]:
    """Per-class decision thresholds written by ``calibrate_thresholds.py``.

    Returns None when the sidecar is absent — serving then falls back to 0.5,
    which is a defensible rank cut but a poor screening operating point. The
    absence is logged, not hidden.
    """
    sidecar = os.path.join(os.path.dirname(checkpoint_path), "thresholds.json")
    try:
        with open(sidecar) as handle:
            data = json.load(handle)
        if list(data.get("classes", [])) != list(FINDINGS):
            raise ValueError("thresholds.json class order does not match serving order")
        per_class = data["per_class"]
        return [float(per_class[name]["threshold"]) for name in FINDINGS]
    except FileNotFoundError:
        logger.warning("No thresholds.json beside the checkpoint — using 0.5 for every class")
        return None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("thresholds.json unusable (%s) — using 0.5", exc)
        return None


class LungClassificationPipeline:
    """Lazily-loaded inference wrapper; the router owns the singleton."""

    def __init__(self, checkpoint_path: str, device: Optional[str] = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: Optional[torch.nn.Module] = None
        self.load_time_ms: Optional[float] = None
        self.checkpoint_epoch: Optional[int] = None
        self.checkpoint_kind: Optional[str] = None
        self.thresholds: Optional[list[float]] = load_thresholds(checkpoint_path)

    # ── lifecycle ──────────────────────────────────────────────────────────

    def load(self) -> None:
        if self.model is not None:
            return
        started = time.perf_counter()
        cache_dir = os.path.join(os.path.dirname(self.checkpoint_path), "xrv")
        model = build_model(cache_dir=cache_dir)

        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        if isinstance(payload, dict):
            self.checkpoint_epoch = payload.get("epoch")
            self.checkpoint_kind = payload.get("kind")
            classes = payload.get("classes")
            if classes and tuple(classes) != FINDINGS:
                raise RuntimeError(
                    f"Checkpoint class order {classes} does not match the serving "
                    f"order {list(FINDINGS)}; refusing to serve misaligned outputs."
                )
        model.load_state_dict(state, strict=True)
        model.eval().to(self.device)
        self.model = model
        self.load_time_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "XRV-16 loaded in %.0f ms on %s (epoch %s, %s)",
            self.load_time_ms,
            self.device,
            self.checkpoint_epoch,
            self.checkpoint_kind,
        )

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
        """Grayscale 224x224, scaled to XRV's [-1024, 1024] — nothing else."""
        gray = image.convert("L").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        array = np.asarray(gray, dtype=np.float32) / 255.0
        array = array * 2048.0 - 1024.0
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
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

        with GradCAM(self.model) as cam_gen:
            cam = cam_gen.generate(tensor.clone(), top_index)
        rgb = np.asarray(image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE)), dtype=np.uint8)
        overlay = overlay_cam_on_image(rgb, cam, alpha=0.42)
        buffer = io.BytesIO()
        Image.fromarray(overlay).save(buffer, format="PNG", optimize=True)
        gradcam_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        thresholds = self.thresholds or [DEFAULT_THRESHOLD] * len(FINDINGS)
        positives = [bool(probabilities[i] >= thresholds[i]) for i in range(len(FINDINGS))]

        bbox = cam_to_bbox(cam, threshold=0.6)
        return {
            "top_finding": FINDINGS[top_index],
            "confidence": float(probabilities[top_index]),
            "probabilities": [float(p) for p in probabilities],
            "thresholds": [float(t) for t in thresholds],
            "positives": positives,
            "top5": top5,
            "gradcam_base64": gradcam_base64,
            "gradcam_grid": [INPUT_SIZE, INPUT_SIZE],
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
            "epoch": self.checkpoint_epoch,
            "kind": self.checkpoint_kind,
            "thresholds_calibrated": self.thresholds is not None,
        }


__all__ = ["LungClassificationPipeline", "FINDINGS", "FINDING_INDEX"]
