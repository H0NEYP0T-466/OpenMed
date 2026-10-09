"""FastAPI router for the MedNeXt lesion segmenter.

Endpoints
---------
POST /api/brain/analyze            classify; if the class is not Normal, segment
GET  /api/brain/mednext-health     readiness probe for the segmenter
GET  /api/brain/mednext-model-info checkpoint metadata and held-out metrics

``/analyze`` is what the workspace's Run button calls. The classifier always
answers. Segmentation is attached only when the predicted class is not
``Normal``, and a missing or unusable MedNeXt checkpoint degrades to a
classification-only response that says why, instead of failing the whole run.

The LiteMedSAM assistive endpoints (``/segment-click``, ``/segment-box``) live in
:mod:`..experimental_lab` and are not touched here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Literal, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...classification.brain_regions import LOCALIZATION_BASIS, map_prediction_to_3d
from ...classification.router import (
    MAX_UPLOAD_BYTES,
    ClassificationResponse,
    _get_pipeline,
    _validate_upload,
)
from .pipeline import (
    MedNeXtSegmenter,
    SegmentationUnavailableError,
    default_checkpoint_path,
    encode_mask_png,
    encode_overlay,
    load_serving_checkpoint,
)

logger = logging.getLogger(__name__)

router = APIRouter()

NORMAL_CLASS = "normal"

_segmenter: Optional[MedNeXtSegmenter] = None
_segmenter_error: Optional[str] = None
_segmenter_error_key: Optional[tuple[float, int]] = None
_load_lock = asyncio.Lock()


def segmentation_checkpoint_path() -> str:
    return default_checkpoint_path()


def _identity(path: str) -> Optional[tuple[float, int]]:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (stat.st_mtime, stat.st_size)


async def _get_segmenter() -> MedNeXtSegmenter:
    """Lazily build the singleton. Raises :class:`SegmentationUnavailableError`."""
    global _segmenter, _segmenter_error, _segmenter_error_key

    if _segmenter is not None:
        return _segmenter

    path = segmentation_checkpoint_path()
    identity = _identity(path)
    if _segmenter_error is not None:
        if _segmenter_error_key == identity:
            raise SegmentationUnavailableError(_segmenter_error)
        _segmenter_error = None
        _segmenter_error_key = None
        logger.info("MedNeXt checkpoint changed since the last failure; retrying load.")

    async with _load_lock:
        if _segmenter is not None:
            return _segmenter
        started = time.perf_counter()
        try:
            _segmenter = await run_in_threadpool(lambda: MedNeXtSegmenter(path))
        except SegmentationUnavailableError as exc:
            _segmenter_error, _segmenter_error_key = str(exc), identity
            logger.error("MedNeXt unavailable: %s", exc)
            raise
        logger.info("MedNeXt ready in %.2fs", time.perf_counter() - started)
        return _segmenter


def _is_normal(result: dict[str, Any]) -> bool:
    tumor_type = str(result.get("tumor_type", "")).strip().lower()
    predicted = str(result.get("predicted_class", "")).strip().lower()
    return tumor_type == NORMAL_CLASS or predicted.startswith(NORMAL_CLASS)


class AnalyzeResponse(ClassificationResponse):
    """Classification fields plus the segmentation outcome."""

    segmentation_performed: bool
    segmentation_skipped_reason: Optional[str] = None
    segmentation_model: Optional[str] = None
    seg_mask_base64: Optional[str] = None
    seg_overlay_base64: Optional[str] = None
    mask_foreground_px: Optional[int] = None
    mask_foreground_fraction: Optional[float] = None
    mean_probability: Optional[float] = None
    peak_probability: Optional[float] = None
    n_components: Optional[int] = None
    box_coords: Optional[list[int]] = None
    segmentation_input_size: Optional[str] = None
    original_size: Optional[str] = None
    segmentation_threshold: Optional[float] = None
    segmentation_tta: Optional[bool] = None
    segmentation_ms: Optional[float] = None
    total_ms: float = 0.0


class MedNeXtHealth(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: Literal["ok", "unavailable"]
    model_loaded: bool
    checkpoint_present: bool
    detail: Optional[str] = None


class MedNeXtInfo(BaseModel):
    available: bool
    detail: Optional[str] = None
    info: dict[str, Any] = Field(default_factory=dict)


def _skipped(reason: str) -> dict[str, Any]:
    return {"segmentation_performed": False, "segmentation_skipped_reason": reason}


async def _segment(payload: bytes) -> dict[str, Any]:
    try:
        segmenter = await _get_segmenter()
    except SegmentationUnavailableError as exc:
        return _skipped(f"MedNeXt segmenter unavailable: {exc}")

    try:
        result, gray = await run_in_threadpool(segmenter.predict_bytes, payload)
    except ValueError as exc:
        return _skipped(f"Segmentation could not read the image: {exc}")
    except RuntimeError as exc:
        logger.error("MedNeXt inference failed: %s", exc, exc_info=True)
        return _skipped("Segmentation failed during inference; the classification is unaffected.")

    height, width = result.original_size
    return {
        "segmentation_performed": True,
        "segmentation_model": f"MedNeXt-{segmenter.spec.variant} (2D)",
        "seg_mask_base64": encode_mask_png(result.mask),
        "seg_overlay_base64": encode_overlay(gray, result.mask),
        "mask_foreground_px": result.foreground_px,
        "mask_foreground_fraction": result.foreground_fraction,
        "mean_probability": result.mean_probability,
        "peak_probability": result.peak_probability,
        "n_components": result.n_components,
        "box_coords": result.bbox,
        "segmentation_input_size": f"{result.input_size}x{result.input_size}",
        "original_size": f"{width}x{height}",
        "segmentation_threshold": result.threshold,
        "segmentation_tta": result.tta,
        "segmentation_ms": round(result.inference_ms, 1),
    }


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_brain_scan(file: UploadFile = File(...)) -> AnalyzeResponse:
    """Classify the scan, then segment the lesion unless it is classified Normal."""
    started = time.perf_counter()
    logger.info("Analyze request file=%s content_type=%s", file.filename, file.content_type)

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    image = _validate_upload(file, payload)
    classifier = await _get_pipeline()

    try:
        result = await run_in_threadpool(classifier.predict_with_gradcam, image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Inference failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail="Inference failed while processing the image."
        ) from exc

    result["locations_3d"] = map_prediction_to_3d(
        result["predicted_class"], locations=None, model_confidence=result["confidence"],
    )
    result["localization_basis"] = LOCALIZATION_BASIS
    result["inference_ms"] = round((time.perf_counter() - started) * 1000.0, 1)

    if _is_normal(result):
        logger.info("Classified Normal — segmentation skipped.")
        segmentation = _skipped("Classified as Normal — there is no lesion to segment.")
    else:
        segmentation = await _segment(payload)

    total_ms = round((time.perf_counter() - started) * 1000.0, 1)
    logger.info(
        "Analyze class=%s conf=%.3f segmented=%s total=%.0fms",
        result["predicted_class"], result["confidence"],
        segmentation["segmentation_performed"], total_ms,
    )
    return AnalyzeResponse(**result, **segmentation, total_ms=total_ms)


@router.get("/mednext-health", response_model=MedNeXtHealth)
async def mednext_health() -> MedNeXtHealth:
    path = segmentation_checkpoint_path()
    present = os.path.isfile(path)
    if _segmenter is not None:
        return MedNeXtHealth(status="ok", model_loaded=True, checkpoint_present=True)
    if _segmenter_error is not None and _segmenter_error_key == _identity(path):
        return MedNeXtHealth(
            status="unavailable", model_loaded=False, checkpoint_present=present,
            detail=_segmenter_error,
        )
    if not present:
        return MedNeXtHealth(
            status="unavailable", model_loaded=False, checkpoint_present=False,
            detail=f"No MedNeXt checkpoint at {path}. Place mednext_brain_seg.pth there.",
        )
    return MedNeXtHealth(
        status="ok", model_loaded=False, checkpoint_present=True,
        detail="Checkpoint present; the network loads on the first analysis.",
    )


@router.get("/mednext-model-info", response_model=MedNeXtInfo)
async def mednext_model_info() -> MedNeXtInfo:
    if _segmenter is not None:
        return MedNeXtInfo(available=True, info=_segmenter.model_info())
    try:
        payload = await run_in_threadpool(load_serving_checkpoint, segmentation_checkpoint_path())
    except SegmentationUnavailableError as exc:
        return MedNeXtInfo(available=False, detail=str(exc))
    return MedNeXtInfo(
        available=True,
        info={
            "architecture": payload["spec"], "input_size": payload["preprocess"]["image_size"],
            "postprocess": payload["postprocess"], "metrics": payload["metrics"],
            "training": payload["training"],
        },
    )
