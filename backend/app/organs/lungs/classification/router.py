"""Lung classification API — LungV2S (4-class) + stock XRV DenseNet-121 (18-label).

Mounted at ``/api/lungs``.

Endpoints
    POST /classify      multipart image -> both models, per-model failure isolated
    GET  /health        per-model residency and checkpoint presence
    GET  /model-info    per-model tag, class list, input size, checkpoint metadata
    POST /warmup        load the models into memory (called when the section opens)
    POST /unload        release them (called when the section closes)

Nothing is constructed or imported eagerly. Importing this module must not pull
``torchxrayvision`` and must not touch a weight file; the health endpoint works
on a cold process with no checkpoint on disk.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from typing import Any, Literal, Optional

from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from .pipeline import DualLungPipeline, current_rss_mb

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"RIFF")

_facade: Optional[DualLungPipeline] = None
# One lock guards facade creation and every load, so two concurrent requests
# cannot race two 100 MB weight loads into the same process.
_load_lock = asyncio.Lock()

ModelName = Literal["v2s", "xrv", "all"]


def checkpoints_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def v2s_checkpoint_path() -> str:
    configured = os.getenv("OPENMED_LUNGS_V2S_CHECKPOINT")
    if configured:
        return os.path.expanduser(configured)
    return os.path.join(checkpoints_dir(), "lung_v2s_best.pth")


def xrv_cache_dir() -> str:
    configured = os.getenv("OPENMED_LUNGS_XRV_CACHE")
    if configured:
        return os.path.expanduser(configured)
    return os.path.join(checkpoints_dir(), "xrv")


def get_facade() -> DualLungPipeline:
    """Return the process-wide facade, building the (weight-free) shell once."""
    global _facade
    if _facade is None:
        _facade = DualLungPipeline(
            v2s_checkpoint=v2s_checkpoint_path(),
            xrv_cache_dir=xrv_cache_dir(),
        )
    return _facade


class ModelSelection(BaseModel):
    model: ModelName = "all"


class ModelHealth(BaseModel):
    loaded: bool
    checkpoint_present: bool
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    process_rss_mb: Optional[float] = None
    v2s: ModelHealth
    xrv: ModelHealth


class ModelInfoResponse(BaseModel):
    v2s: dict[str, Any]
    xrv: dict[str, Any]


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


async def _load(which: str) -> dict[str, Optional[str]]:
    """Load under the shared lock; the blocking work runs off the event loop."""
    facade = get_facade()
    async with _load_lock:
        return await asyncio.to_thread(facade.load, which)


@router.post("/classify")
async def classify_lung_scan(
    file: UploadFile = File(...),
    include_xrv_detail: bool = Query(
        False,
        description="Include all 18 XRV scores and the XRV Grad-CAM (second, on-demand call).",
    ),
) -> dict[str, Any]:
    payload = await file.read()
    image = _validate_upload(file, payload)

    facade = get_facade()
    await _load("all")

    result = await asyncio.to_thread(facade.classify, image, include_xrv_detail)

    if result["v2s"] is None and result["xrv"] is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Both lung models failed.",
                "v2s": result["errors"]["v2s"],
                "xrv": result["errors"]["xrv"],
            },
        )
    return result


@router.post("/xrv-cam")
async def xrv_gradcam(file: UploadFile = File(...)) -> dict[str, Any]:
    """XRV Grad-CAM only.

    The 18 scores already ship with ``/classify``, so opening the detail panel
    must not re-run both models. This runs XRV alone and returns just the map.
    """
    payload = await file.read()
    image = _validate_upload(file, payload)
    await _load("xrv")
    result = await asyncio.to_thread(get_facade().xrv.predict, image, True)
    if not result.get("gradcam_base64"):
        raise HTTPException(status_code=503, detail="XRV Grad-CAM could not be produced.")
    return {
        "gradcam_base64": result["gradcam_base64"],
        "inference_ms": result["inference_ms"],
    }


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    # Reads file presence and in-memory flags only — never builds a model, so it
    # answers on a cold process with an empty checkpoints directory.
    return HealthResponse(**get_facade().health())


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**get_facade().model_info())


@router.post("/warmup")
async def warmup_lung_classifier(
    payload: Optional[ModelSelection] = Body(default=None),
) -> dict[str, Any]:
    which = (payload or ModelSelection()).model
    errors = await _load(which)
    facade = get_facade()
    loaded = {"v2s": facade.v2s.is_loaded, "xrv": facade.xrv.is_loaded}
    failed = [name for name, error in errors.items() if error]
    if failed and len(failed) == len(errors):
        raise HTTPException(
            status_code=503,
            detail={"message": "No lung model could be loaded.", "errors": errors},
        )
    return {
        "status": "ok" if not failed else "partial",
        "requested": which,
        "loaded": loaded,
        "errors": errors,
        "device": str(facade.v2s.device),
    }


@router.post("/unload")
async def unload_lung_classifier(
    payload: Optional[ModelSelection] = Body(default=None),
) -> dict[str, Any]:
    which = (payload or ModelSelection()).model
    facade = get_facade()
    async with _load_lock:
        unloaded = await asyncio.to_thread(facade.unload, which)
    if any(unloaded.values()):
        logger.info(
            "Lung models unloaded (%s), memory freed. RSS=%s MB",
            which,
            current_rss_mb(),
        )
    return {
        "status": "ok",
        "requested": which,
        "unloaded": unloaded,
        "message": "Lung models released; memory freed.",
    }
