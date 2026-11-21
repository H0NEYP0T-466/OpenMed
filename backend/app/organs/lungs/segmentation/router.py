"""Lungs segmentation API: lung field plus COVID opacity segmentation.

Mounted at ``/api/lungs`` beside the classification router (no path overlaps).

Endpoints
    POST /segment          multipart image -> lung field, COVID opacity, merged label map
    GET  /seg-health       checkpoint presence and residency (never builds a model)
    GET  /seg-model-info   checkpoint metadata and the held-out metrics from the sidecar
    POST /seg-warmup       load the segmenter (called when the lungs section opens)
    POST /seg-unload       release it (called when the section closes)

Failures are typed: ``{"detail": {"code": ..., "message": ...}}``. A missing or
unusable checkpoint degrades to a typed 503 that says why; it never takes the
process down, and importing this module touches neither weights nor
``transformers``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Literal

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .checkpoint import (
    CHECKPOINT_NAME,
    SegmentationUnavailableError,
    default_checkpoint_path,
    load_serving_checkpoint,
)
from .labels import LEGEND, SCOPE_NOTICE, TASK_NAME
from .model import MODEL_TAG
from .pipeline import (
    LungSegmenter,
    SegResult,
    current_rss_mb,
    encode_label_map_png,
    encode_mask_png,
    info_from_payload,
    output_size,
)
from .preprocessor import decode_gray

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_SIDE_PX = 4096
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"RIFF")

_segmenter: LungSegmenter | None = None
_segmenter_error: str | None = None
_segmenter_error_code: str = "checkpoint_unusable"
_segmenter_error_key: tuple[float, int] | None = None
# One lock guards creation, unload and every load, so two concurrent requests
# cannot race two weight loads into the same process.
_load_lock = asyncio.Lock()


def segmentation_checkpoint_path() -> str:
    return default_checkpoint_path()


def _identity(path: str) -> tuple[float, int] | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (stat.st_mtime, stat.st_size)


def _typed(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


# ------------------------------------------------------------------------- models


class _Model(BaseModel):
    model_config = {"protected_namespaces": ()}


class SegHealth(_Model):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    checkpoint_present: bool
    detail: str | None = None


class SegWarmup(SegHealth):
    device: str | None = None


class SegUnload(_Model):
    status: Literal["ok"] = "ok"
    unloaded: bool
    message: str


class SegModelInfo(_Model):
    available: bool
    detail: str | None = None
    info: dict[str, Any] = Field(default_factory=dict)


class Size(_Model):
    width: int
    height: int


class LegendItem(_Model):
    value: int
    name: str


class SegStats(_Model):
    lung_area_px: int
    lung_fraction_of_frame: float
    lung_components: int
    lung_bbox: list[int] | None = None
    opacity_area_px: int | None = None
    opacity_fraction_of_lung: float | None = None
    opacity_components: int | None = None
    opacity_peak_probability: float | None = None
    opacity_mean_probability: float | None = None
    crop_box: list[int] | None = None
    crop_fallback: bool = False


class SegResponse(_Model):
    task: str
    model_tag: str
    opacity_available: bool
    opacity_unavailable_reason: str | None = None
    image_size: Size
    mask_size: Size
    lung_mask_png: str
    opacity_mask_png: str | None = None
    label_map_png: str
    legend: list[LegendItem]
    stats: SegStats
    postprocess: dict[str, Any]
    inference_ms: float
    scope_notice: str


# ---------------------------------------------------------------------- residency


async def _get_segmenter() -> LungSegmenter:
    """Lazily build the singleton. Raises :class:`SegmentationUnavailableError`."""
    global _segmenter, _segmenter_error, _segmenter_error_code, _segmenter_error_key

    if _segmenter is not None and _segmenter.is_loaded:
        return _segmenter

    path = segmentation_checkpoint_path()
    identity = _identity(path)
    if _segmenter_error is not None:
        if _segmenter_error_key == identity:
            raise SegmentationUnavailableError(_segmenter_error, _segmenter_error_code)
        _segmenter_error, _segmenter_error_key = None, None
        logger.info("Lungs segmentation checkpoint changed since the last failure; retrying load.")

    async with _load_lock:
        if _segmenter is not None and _segmenter.is_loaded:
            return _segmenter
        started = time.perf_counter()
        try:
            _segmenter = await run_in_threadpool(LungSegmenter, path)
        except SegmentationUnavailableError as exc:
            _segmenter_error, _segmenter_error_key = str(exc), identity
            _segmenter_error_code = exc.code
            logger.error("Lungs segmenter unavailable: %s", exc)
            raise
        except Exception as exc:  # noqa: BLE001 - anything else is also fail-closed, with the reason
            message = f"Lungs segmenter failed to load: {type(exc).__name__}: {exc}"
            _segmenter_error, _segmenter_error_key = message, identity
            _segmenter_error_code = "checkpoint_unusable"
            logger.error(message, exc_info=True)
            raise SegmentationUnavailableError(message) from exc
        logger.info("Lungs segmenter ready in %.2fs", time.perf_counter() - started)
        return _segmenter


def _health_snapshot() -> SegHealth:
    path = segmentation_checkpoint_path()
    present = os.path.isfile(path)
    if _segmenter is not None and _segmenter.is_loaded:
        return SegHealth(status="ok", model_loaded=True, checkpoint_present=True,
                         detail=f"Resident on {_segmenter.device}.")
    if _segmenter_error is not None and _segmenter_error_key == _identity(path):
        return SegHealth(status="unavailable", model_loaded=False, checkpoint_present=present,
                         detail=_segmenter_error)
    if not present:
        return SegHealth(
            status="unavailable", model_loaded=False, checkpoint_present=False,
            detail=f"No lungs segmentation checkpoint at {path}. Place {CHECKPOINT_NAME} and its "
                   "sidecar there.",
        )
    return SegHealth(status="ok", model_loaded=False, checkpoint_present=True,
                     detail="Checkpoint present; the network loads on first use.")


# ------------------------------------------------------------------------- upload


def _validate_upload(file: UploadFile, payload: bytes) -> None:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise _typed(415, "unsupported_media",
                     f"Unsupported content type {file.content_type!r}; upload a JPEG or PNG.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise _typed(413, "payload_too_large", "Image exceeds the 12 MB limit.")
    if not payload.startswith(ALLOWED_MAGIC):
        raise _typed(415, "unsupported_media", "File signature is not a supported image.")


def _decode(payload: bytes) -> Any:
    try:
        gray = decode_gray(payload)
    except ValueError as exc:
        raise _typed(415, "invalid_image", "Image could not be decoded.") from exc
    if max(gray.shape) > MAX_SIDE_PX:
        raise _typed(413, "payload_too_large",
                     f"Image is {gray.shape[1]}x{gray.shape[0]} px; the limit is {MAX_SIDE_PX} px per side.")
    return gray


def _build_response(result: SegResult, segmenter: LungSegmenter) -> SegResponse:
    height, width = result.original_size
    mask_h, mask_w = output_size(result.lung_mask.shape)
    post: dict[str, Any] = {
        "lung": result.lung_post.to_dict(),
        "opacity": result.opacity_post.to_dict() if result.opacity_post else None,
    }
    return SegResponse(
        task=TASK_NAME,
        model_tag=MODEL_TAG,
        opacity_available=result.opacity_available,
        opacity_unavailable_reason=result.opacity_unavailable_reason,
        image_size=Size(width=width, height=height),
        mask_size=Size(width=mask_w, height=mask_h),
        lung_mask_png=encode_mask_png(result.lung_mask),
        opacity_mask_png=(
            encode_mask_png(result.opacity_mask) if result.opacity_mask is not None else None
        ),
        label_map_png=encode_label_map_png(result.label_map),
        legend=[LegendItem(**item) for item in LEGEND],
        stats=SegStats(**result.stats),
        postprocess=post,
        inference_ms=round(result.inference_ms, 1),
        scope_notice=SCOPE_NOTICE,
    )


# ------------------------------------------------------------------------ routes


@router.post("/segment", response_model=SegResponse)
async def segment_lung_scan(file: UploadFile = File(...)) -> SegResponse:
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    _validate_upload(file, payload)
    gray = _decode(payload)

    try:
        segmenter = await _get_segmenter()
    except SegmentationUnavailableError as exc:
        raise _typed(503, exc.code, str(exc)) from exc

    try:
        result = await run_in_threadpool(segmenter.predict_gray, gray)
    except SegmentationUnavailableError as exc:
        raise _typed(503, exc.code, str(exc)) from exc
    except (RuntimeError, ValueError, cv2.error) as exc:
        logger.error("Lungs segmentation failed: %s", exc, exc_info=True)
        raise _typed(500, "inference_failed",
                     "Segmentation failed while processing the image.") from exc
    return _build_response(result, segmenter)


@router.get("/seg-health", response_model=SegHealth)
async def seg_health() -> SegHealth:
    # File presence and in-memory flags only; never builds a model.
    return _health_snapshot()


@router.get("/seg-model-info", response_model=SegModelInfo)
async def seg_model_info() -> SegModelInfo:
    if _segmenter is not None and _segmenter.is_loaded:
        return SegModelInfo(available=True, info=_segmenter.model_info())
    path = segmentation_checkpoint_path()
    try:
        payload = await run_in_threadpool(load_serving_checkpoint, path)
    except SegmentationUnavailableError as exc:
        return SegModelInfo(available=False, detail=str(exc))
    return SegModelInfo(
        available=True,
        info=info_from_payload(payload, path, loaded=False, device=None),
    )


@router.post("/seg-warmup", response_model=SegWarmup)
async def seg_warmup() -> SegWarmup:
    """Load the segmenter ahead of the first request. Always answers 200."""
    try:
        segmenter = await _get_segmenter()
    except SegmentationUnavailableError as exc:
        present = os.path.isfile(segmentation_checkpoint_path())
        return SegWarmup(status="unavailable", model_loaded=False,
                         checkpoint_present=present, detail=str(exc))
    return SegWarmup(
        status="ok", model_loaded=True, checkpoint_present=True,
        detail=f"Lungs segmenter ready on {segmenter.device}.", device=str(segmenter.device),
    )


@router.post("/seg-unload", response_model=SegUnload)
async def seg_unload() -> SegUnload:
    """Release the segmenter and its memory when the user leaves the section."""
    global _segmenter, _segmenter_error, _segmenter_error_key
    async with _load_lock:
        segmenter, _segmenter = _segmenter, None
        _segmenter_error, _segmenter_error_key = None, None
        if segmenter is not None:
            await run_in_threadpool(segmenter.unload)
    freed = segmenter is not None
    # Always logged, resident or not: the terminal has to answer "did it unload?"
    # without the reader having to guess whether silence meant "nothing to do".
    logger.info(
        "Lungs segmenter release request: released=%s, RSS=%s MB",
        freed,
        current_rss_mb(),
    )
    return SegUnload(
        unloaded=freed,
        message="Lungs segmenter released; memory freed." if freed else "Lungs segmenter was not resident.",
    )
