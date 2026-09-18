"""
FastAPI router for brain tumor classification.

Endpoints
---------
POST /api/brain/classify   — Upload an MRI image → classification + Grad-CAM
GET  /api/brain/model-info  — Model metadata (architecture, classes, etc.)
GET  /api/brain/health      — Readiness probe (is the model loaded?)
"""

from __future__ import annotations

import io
import time
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

from .pipeline import BrainClassificationPipeline, CLASS_NAMES
from .brain_regions import map_prediction_to_3d

import os

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Lazy-loaded singleton ────────────────────────────────────────────────
_pipeline: Optional[BrainClassificationPipeline] = None


def _get_pipeline() -> BrainClassificationPipeline:
    """Lazy-load the pipeline on first inference request."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    t0 = time.perf_counter()
    logger.info("🧠  Loading Brain Classification Model …")

    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    model_path = os.path.join(checkpoint_dir, "brain_best_model.pth")

    if os.path.isfile(model_path):
        _pipeline = BrainClassificationPipeline(model_path=model_path)
        dt = time.perf_counter() - t0
        logger.info(f"✅  Model loaded in {dt:.2f}s  checkpoint={model_path}")
    else:
        # Fall back to pretrained weights (predictions will be random)
        logger.warning(
            f"⚠️  No checkpoint found at {model_path} — "
            "loading ImageNet-pretrained weights (inference will be inaccurate)."
        )
        _pipeline = BrainClassificationPipeline()
        dt = time.perf_counter() - t0
        logger.info(f"✅  Model initialised (no trained checkpoint) in {dt:.2f}s")

    return _pipeline


# ── POST /classify ───────────────────────────────────────────────────────

@router.post("/classify")
async def classify_brain_tumor(file: UploadFile = File(...)):
    """
    Upload a brain MRI image (JPEG/PNG) and receive a classification result
    with Grad-CAM overlay and 3D brain-region mapping.
    """
    t0 = time.perf_counter()
    logger.info(f"📥  Inference request  file={file.filename}  content_type={file.content_type}")

    pipeline = _get_pipeline()

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.info(f"   Image decoded  size={image.size}  mode={image.mode}")
    except Exception as exc:
        logger.error(f"❌  Image decode failed: {exc}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    try:
        result = pipeline.predict_with_gradcam(image)

        # Enrich with 3D location mapping
        result["locations_3d"] = map_prediction_to_3d(
            result["predicted_class"], locations=None
        )

        dt = time.perf_counter() - t0
        logger.info(
            f"✅  Prediction  class={result['predicted_class']}  "
            f"conf={result['confidence']:.3f}  "
            f"tumor_type={result['tumor_type']}  "
            f"time={dt:.3f}s"
        )
        return result

    except Exception as exc:
        logger.error(f"❌  Inference failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")


# ── GET /model-info ──────────────────────────────────────────────────────

@router.get("/model-info")
async def get_model_info():
    """Return model metadata."""
    return {
        "model_name": "EfficientNetV2-B2",
        "model_tag": "tf_efficientnetv2_b2.in1k",
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "input_size": "260×260 (from timm config)",
        "task": "39-class Brain Tumor Classification",
        "dataset": "12,626 MRI scans · 13 tumor types · 3 MRI sequences",
    }


# ── GET /health ──────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """Readiness probe — returns whether the model is currently loaded."""
    return {
        "status": "ok",
        "model_loaded": _pipeline is not None,
    }
