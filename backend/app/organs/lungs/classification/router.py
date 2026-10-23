"""Lung classification API — Z-Net on NIH ChestX-ray14.

Endpoints (mounted at /api/lungs):
  POST /classify      multipart image -> 14 probabilities + Grad-CAM overlay
  GET  /health        checkpoint presence and residency
  GET  /model-info    label space and training metadata
  POST /warmup        load the model into memory (called when the section opens)
  POST /unload        release it (called when the section closes)
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from typing import Any, Literal, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from .model import FINDINGS, MODEL_TAG
from .pipeline import LungClassificationPipeline

logger = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_CHECKPOINT = "lung_best_model.pth"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"RIFF")

_pipeline: Optional[LungClassificationPipeline] = None
_pipeline_error: Optional[str] = None
_load_lock = asyncio.Lock()


def checkpoint_path() -> str:
    configured = os.getenv("OPENMED_LUNGS_CHECKPOINT")
    if configured:
        return os.path.expanduser(configured)
    return os.path.join(os.path.dirname(__file__), "checkpoints", DEFAULT_CHECKPOINT)


class Top5Entry(BaseModel):
    finding: str
    probability: float
    rank: int


class ClassificationResponse(BaseModel):
    top_finding: str
    confidence: float
    probabilities: list[float]
    top5: list[Top5Entry]
    gradcam_base64: str
    gradcam_grid: list[int]
    bbox: Optional[list[int]] = None
    inference_ms: float
    model_tag: str


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    checkpoint_present: bool
    detail: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_tag: str
    findings: list[str]
    paper: str
    input_size: int
    checkpoint: dict[str, Any]


async def _get_pipeline() -> LungClassificationPipeline:
    global _pipeline, _pipeline_error
    if _pipeline is not None:
        return _pipeline
    async with _load_lock:
        if _pipeline is not None:
            return _pipeline
        path = checkpoint_path()
        if not os.path.isfile(path):
            _pipeline_error = f"Checkpoint not found at {path}"
            raise HTTPException(status_code=503, detail=_pipeline_error)
        try:
            pipeline = LungClassificationPipeline(path)
            await asyncio.to_thread(pipeline.load)
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            _pipeline_error = f"Failed to load Z-Net: {exc}"
            logger.exception("Z-Net load failed")
            raise HTTPException(status_code=503, detail=_pipeline_error) from exc
        _pipeline = pipeline
        _pipeline_error = None
        return _pipeline


def _validate_upload(file: UploadFile, payload: bytes) -> Image.Image:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type {file.content_type!r}; upload a JPEG or PNG.",
        )
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 12 MB limit.")
    if not payload.startswith(ALLOWED_MAGIC):
        raise HTTPException(status_code=415, detail="File signature is not a supported image.")
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=415, detail="Image could not be decoded.") from exc
    return image


@router.post("/warmup")
async def warmup_lung_classifier() -> dict[str, Any]:
    pipeline = await _get_pipeline()
    return {
        "status": "ok",
        "message": "Z-Net lung classifier loaded.",
        "device": str(pipeline.device),
    }


@router.post("/unload")
async def unload_lung_classifier() -> dict[str, Any]:
    global _pipeline
    if _pipeline is not None:
        _pipeline.unload()
        _pipeline = None
    return {"status": "ok", "message": "Z-Net lung classifier unloaded. Memory freed."}


@router.post("/classify", response_model=ClassificationResponse)
async def classify_lung_scan(file: UploadFile = File(...)) -> ClassificationResponse:
    payload = await file.read()
    image = _validate_upload(file, payload)
    pipeline = await _get_pipeline()
    try:
        result = await asyncio.to_thread(pipeline.predict, image)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Lung inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    return ClassificationResponse(**result)


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info() -> ModelInfoResponse:
    pipeline = _pipeline or LungClassificationPipeline(checkpoint_path())
    return ModelInfoResponse(
        model_tag=MODEL_TAG,
        findings=list(FINDINGS),
        paper="Ahmad et al., Diagnostics 2023, 13(22), 3462",
        input_size=224,
        checkpoint=pipeline.checkpoint_info(),
    )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    present = os.path.isfile(checkpoint_path())
    if not present:
        return HealthResponse(
            status="unavailable",
            model_loaded=False,
            checkpoint_present=False,
            detail=f"Checkpoint missing at {checkpoint_path()}",
        )
    return HealthResponse(
        status="ok",
        model_loaded=_pipeline is not None and _pipeline.is_loaded,
        checkpoint_present=True,
        detail=_pipeline_error,
    )
