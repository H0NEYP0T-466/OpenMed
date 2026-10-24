"""Lifecycle and inference for the TorchXRayVision instrument.

The model is never loaded at import time. `load()` is called when the lab tab
opens; `unload()` drops every reference, empties the CUDA cache and forces a
collection so an idle tab costs nothing.
"""

from __future__ import annotations

import base64
import gc
import io
import logging
import time
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from .gradcam import GradCAM, overlay
from .model import load_model, pathology_names, weights_present

logger = logging.getLogger(__name__)

try:  # optional dependency — the API reports 503 until it is installed
    import torchxrayvision as xrv
except ImportError:  # pragma: no cover
    xrv = None

_model: Optional[Any] = None
_pathologies: list[str] = []
_loaded_at: Optional[float] = None
_load_ms: Optional[float] = None


class XrvUnavailable(RuntimeError):
    """Raised when torchxrayvision is not importable in this environment."""


def is_installed() -> bool:
    return xrv is not None


def is_loaded() -> bool:
    return _model is not None


def status() -> dict[str, Any]:
    return {
        "installed": is_installed(),
        "loaded": is_loaded(),
        "weights_present": weights_present(),
        "pathologies": list(_pathologies),
        "loaded_at": _loaded_at,
        "load_ms": _load_ms,
    }


def load() -> dict[str, Any]:
    global _model, _pathologies, _loaded_at, _load_ms

    if not is_installed():
        raise XrvUnavailable("torchxrayvision not installed in environment")
    if _model is not None:
        return status()

    started = time.perf_counter()
    model = load_model()
    _model = model
    _pathologies = pathology_names(model)
    _load_ms = (time.perf_counter() - started) * 1000.0
    _loaded_at = time.time()
    logger.info("TorchXRayVision loaded in %.0f ms (%d pathologies)",
                _load_ms, len(_pathologies))
    return status()


def unload() -> dict[str, Any]:
    global _model, _pathologies, _loaded_at, _load_ms

    if _model is not None:
        del _model
        _model = None
    _pathologies = []
    _loaded_at = None
    _load_ms = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("TorchXRayVision unloaded; memory released")
    return status()


def _preprocess(image: Image.Image) -> torch.Tensor:
    """xrv pipeline: grayscale -> [-1024,1024] -> centre crop -> 224."""
    assert xrv is not None
    from torchvision import transforms

    array = np.asarray(image.convert("L"), dtype=np.float32)
    array = xrv.datasets.normalize(array, 255)
    array = array[None, ...]  # 1, H, W
    pipeline = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    array = pipeline(array)
    tensor = torch.from_numpy(array[None, ...]).float()
    return tensor


@torch.no_grad()
def _logits(tensor: torch.Tensor) -> np.ndarray:
    assert _model is not None
    return _model(tensor)[0].float().cpu().numpy()


def predict(image: Image.Image) -> dict[str, Any]:
    if not is_installed():
        raise XrvUnavailable("torchxrayvision not installed in environment")
    if _model is None:
        load()

    assert _model is not None
    started = time.perf_counter()
    tensor = _preprocess(image)

    raw = _logits(tensor)
    probabilities = 1.0 / (1.0 + np.exp(-raw))  # xrv heads are logits
    ranked = np.argsort(-probabilities)
    top_index = int(ranked[0])

    predictions = [
        {"pathology": _pathologies[i] if i < len(_pathologies) else f"class_{i}",
         "probability": float(probabilities[i])}
        for i in range(len(probabilities))
    ]

    with GradCAM(_model) as cam_gen:
        cam = cam_gen.generate(tensor.clone().requires_grad_(True), top_index)
    rgb = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.uint8)
    heatmap = overlay(rgb, cam)
    buffer = io.BytesIO()
    Image.fromarray(heatmap).save(buffer, format="PNG", optimize=True)

    return {
        "predictions": predictions,
        "top_pathology": predictions[top_index]["pathology"],
        "confidence": float(probabilities[top_index]),
        "gradcam_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "inference_ms": (time.perf_counter() - started) * 1000.0,
        "weights": "densenet121-res224-all",
    }
