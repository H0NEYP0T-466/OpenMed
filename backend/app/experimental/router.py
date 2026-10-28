"""HTTP surface for the Experimental Laboratory.

Every model is reached through the same five endpoints, parameterised by id, so
the frontend never learns a model-specific route. Adding a model to the registry
is enough for it to appear here.

Lifecycle, which the lab's memory contract depends on:

    tab opened   → POST /{model}/load      weights enter memory
    tab left     → POST /{model}/unload    weights leave memory
    first prompt → predict loads lazily if the client forgot

Endpoints (mounted at /api/experimental):
  GET  /models               every registered model and whether it can load
  GET  /{model}/health       readiness for one model
  POST /{model}/load         load, return milliseconds
  POST /{model}/unload       release
  POST /{model}/predict      multipart image + box → mask, overlay, score
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from typing import Any, Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import registry
from .base import PointPrompt, SegPrompt

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_UPLOAD_BYTES = 24 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"RIFF")

_load_locks: dict[str, asyncio.Lock] = {}


def _lock_for(model_id: str) -> asyncio.Lock:
    if model_id not in _load_locks:
        _load_locks[model_id] = asyncio.Lock()
    return _load_locks[model_id]


def _adapter_or_404(model_id: str):
    try:
        return registry.get(model_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model {model_id!r}. Available: {sorted(m['id'] for m in registry.list_models())}",
        ) from exc


class ModelCard(BaseModel):
    id: str
    title: str
    description: str = ""
    input_size: int = 0
    available: bool
    unavailable_reason: Optional[str] = None
    loaded: bool


class ModelsResponse(BaseModel):
    models: list[ModelCard]
    loaded: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model: str
    loaded: bool
    available: bool
    detail: Optional[str] = None


class LoadResponse(BaseModel):
    status: Literal["ok"]
    model: str
    load_ms: float


class PredictResponse(BaseModel):
    model: str
    score: float
    inference_ms: float
    image_size: list[int]
    mask_base64: str
    overlay_base64: str
    prompt_kinds: list[str]
    mask_pixels: int
    mask_fraction: float
    extra: dict[str, Any] = Field(default_factory=dict)


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    models = registry.list_models()
    return ModelsResponse(
        models=[ModelCard(**m) for m in models],
        loaded=registry.loaded_models(),
    )


@router.get("/{model_id}/health", response_model=HealthResponse)
async def health(model_id: str) -> HealthResponse:
    adapter = _adapter_or_404(model_id)
    available, reason = adapter.availability()
    return HealthResponse(
        status="ok" if available else "unavailable",
        model=model_id,
        loaded=adapter.is_loaded,
        available=available,
        detail=reason or None,
    )


@router.post("/{model_id}/load", response_model=LoadResponse)
async def load(model_id: str) -> LoadResponse:
    adapter = _adapter_or_404(model_id)
    async with _lock_for(model_id):
        available, reason = adapter.availability()
        if not available:
            raise HTTPException(status_code=503, detail=reason)
        try:
            load_ms = await asyncio.to_thread(adapter.load)
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            logger.exception("%s failed to load", model_id)
            raise HTTPException(status_code=503, detail=f"{model_id} failed to load: {exc}") from exc
    return LoadResponse(status="ok", model=model_id, load_ms=load_ms)


@router.post("/{model_id}/unload")
async def unload(model_id: str) -> dict[str, Any]:
    _adapter_or_404(model_id)
    released = await asyncio.to_thread(registry.unload, model_id)
    return {"status": "ok", "model": model_id, "released": released}


def _validate_upload(file: UploadFile, payload: bytes) -> Image.Image:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type {file.content_type!r}; upload a JPEG or PNG.",
        )
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 24 MB limit.")
    if not payload.startswith(ALLOWED_MAGIC):
        raise HTTPException(status_code=415, detail="File signature is not a supported image.")
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=415, detail="Image could not be decoded.") from exc
    return image


@router.post("/{model_id}/predict", response_model=PredictResponse)
async def predict(
    model_id: str,
    file: UploadFile = File(...),
    x1: Optional[int] = Form(None),
    y1: Optional[int] = Form(None),
    x2: Optional[int] = Form(None),
    y2: Optional[int] = Form(None),
    points: Optional[str] = Form(None),
    mask: Optional[UploadFile] = File(None),
) -> PredictResponse:
    """Any combination of box, points and mask — SAM2 composes all three.

    `points` is JSON: [[x, y, 1], [x, y, 0], ...] where the third value is 1 for
    foreground and 0 for background. `mask` is a PNG of a previous result, used
    to refine rather than start over.
    """
    adapter = _adapter_or_404(model_id)
    payload = await file.read()
    image = _validate_upload(file, payload)

    box = None
    if None not in (x1, y1, x2, y2):
        box = (int(x1), int(y1), int(x2), int(y2))

    parsed_points: list[PointPrompt] = []
    if points:
        try:
            raw = json.loads(points)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"points is not valid JSON: {exc}") from exc
        if not isinstance(raw, list):
            raise HTTPException(status_code=422, detail="points must be a list of [x, y, label]")
        for item in raw:
            if not isinstance(item, list) or len(item) < 2:
                raise HTTPException(status_code=422, detail=f"bad point {item!r}")
            label = item[2] if len(item) > 2 else 1
            parsed_points.append(PointPrompt(int(item[0]), int(item[1]), bool(label)))

    mask_png = None
    if mask is not None:
        mask_png = await mask.read()
        if not mask_png:
            mask_png = None

    try:
        prompt = SegPrompt(box=box, points=tuple(parsed_points), mask_png=mask_png)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    async with _lock_for(model_id):
        available, reason = adapter.availability()
        if not available:
            raise HTTPException(status_code=503, detail=reason)
        try:
            result = await asyncio.to_thread(adapter.predict, image, prompt)
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s inference failed", model_id)
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictResponse(
        model=result.model,
        score=result.score,
        inference_ms=result.inference_ms,
        image_size=list(result.image_size),
        mask_base64=result.mask_base64,
        overlay_base64=result.overlay_base64,
        prompt_kinds=list(result.extra.get("prompt_kinds", [])),
        mask_pixels=int(result.extra.get("mask_pixels", 0)),
        mask_fraction=float(result.extra.get("mask_fraction", 0.0)),
        extra=result.extra,
    )
