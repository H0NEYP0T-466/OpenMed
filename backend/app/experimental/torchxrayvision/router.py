"""TorchXRayVision instrument API.

  POST /load      load the DenseNet-121 baseline (called when the tab opens)
  POST /unload    release it (called when the tab closes)
  GET  /status    installation, weights and residency
  POST /predict   multipart image -> 18 probabilities + Grad-CAM overlay
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any, Literal, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from . import service

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"RIFF")

_lock = asyncio.Lock()


class StatusResponse(BaseModel):
    installed: bool
    loaded: bool
    weights_present: bool
    pathologies: list[str]
    loaded_at: Optional[float] = None
    load_ms: Optional[float] = None


class Prediction(BaseModel):
    pathology: str
    probability: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    top_pathology: str
    confidence: float
    gradcam_base64: str
    inference_ms: float
    weights: str


def _unavailable() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="torchxrayvision not installed in environment",
    )


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    return StatusResponse(**service.status())


@router.post("/load", response_model=StatusResponse)
async def load_model() -> StatusResponse:
    if not service.is_installed():
        raise _unavailable()
    async with _lock:
        try:
            state = await asyncio.to_thread(service.load)
        except service.XrvUnavailable as exc:
            raise _unavailable() from exc
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            logger.exception("TorchXRayVision load failed")
            raise HTTPException(status_code=500, detail=f"Load failed: {exc}") from exc
    return StatusResponse(**state)


@router.post("/unload", response_model=StatusResponse)
async def unload_model() -> StatusResponse:
    async with _lock:
        state = await asyncio.to_thread(service.unload)
    return StatusResponse(**state)


@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if not service.is_installed():
        raise _unavailable()

    payload = await file.read()
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 12 MB limit.")
    if not payload.startswith(ALLOWED_MAGIC):
        raise HTTPException(status_code=415, detail="File signature is not a supported image.")
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=415, detail="Image could not be decoded.") from exc

    try:
        result: dict[str, Any] = await asyncio.to_thread(service.predict, image)
    except service.XrvUnavailable as exc:
        raise _unavailable() from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("TorchXRayVision inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictResponse(**result)
