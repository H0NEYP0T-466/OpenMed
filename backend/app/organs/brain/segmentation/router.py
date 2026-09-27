"""FastAPI router for brain tumour segmentation.

Endpoints
---------
POST /api/brain/segment        Classify + segment an MRI image
POST /api/brain/load-models    Eagerly load both models into memory
POST /api/brain/unload-models  Release both models from memory
GET  /api/brain/seg-health     Segmentation readiness probe
GET  /api/brain/seg-model-info Segmentation model metadata

Lifecycle
---------
The frontend calls ``/load-models`` when the user enters the brain
section and ``/unload-models`` when they leave.  The ``/segment``
endpoint also lazily loads models if they are not yet in memory.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
from typing import Any, Literal, Optional

import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..classification.brain_regions import LOCALIZATION_BASIS, map_prediction_to_3d
from ..classification.label_space import CLASS_NAMES
from ..classification.pipeline import BrainClassificationPipeline, ModelUnavailableError
from ..classification.router import checkpoint_path as classification_checkpoint_path
from .pipeline import BrainSegmentationPipeline, SegmentationUnavailableError
from .prepare_weights import default_checkpoint_path, verify_weights

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"RIFF")

# Normal class name — when classifier predicts this, segmentation is skipped
NORMAL_CLASS = "Normal"

# ── Singleton state ───────────────────────────────────────────────────────

_classifier: Optional[BrainClassificationPipeline] = None
_segmenter: Optional[BrainSegmentationPipeline] = None
_load_lock = asyncio.Lock()
_load_error: Optional[str] = None


def segmentation_checkpoint_path() -> str:
    return str(default_checkpoint_path())


async def _ensure_models_loaded() -> tuple[BrainClassificationPipeline, BrainSegmentationPipeline]:
    """Load both classifier and segmenter if not already loaded."""
    global _classifier, _segmenter, _load_error

    if _classifier is not None and _segmenter is not None:
        return _classifier, _segmenter

    async with _load_lock:
        if _classifier is not None and _segmenter is not None:
            return _classifier, _segmenter

        started = time.perf_counter()
        logger.info("Loading brain models (classifier + segmenter)...")

        # ── Classifier ────────────────────────────────────────────
        if _classifier is None:
            cls_path = classification_checkpoint_path()
            try:
                _classifier = await run_in_threadpool(
                    lambda: BrainClassificationPipeline(model_path=cls_path)
                )
                logger.info(
                    "Classifier loaded in %.2fs (classes=%d)",
                    time.perf_counter() - started,
                    _classifier.num_classes,
                )
            except ModelUnavailableError as exc:
                _load_error = f"Classifier: {exc}"
                logger.error("Classifier unavailable: %s", exc)
                raise HTTPException(status_code=503, detail=_load_error) from exc

        # ── Segmenter ─────────────────────────────────────────────
        if _segmenter is None:
            seg_path = segmentation_checkpoint_path()
            try:
                seg_start = time.perf_counter()
                _segmenter = await run_in_threadpool(
                    lambda: BrainSegmentationPipeline(model_path=seg_path)
                )
                logger.info(
                    "Segmenter loaded in %.2fs", time.perf_counter() - seg_start,
                )
            except SegmentationUnavailableError as exc:
                _load_error = f"Segmenter: {exc}"
                logger.error("Segmenter unavailable: %s", exc)
                raise HTTPException(status_code=503, detail=_load_error) from exc

        logger.info(
            "Both brain models ready in %.2fs total",
            time.perf_counter() - started,
        )
        _load_error = None
        return _classifier, _segmenter


async def _unload_models() -> None:
    """Release both models from memory."""
    global _classifier, _segmenter, _load_error

    async with _load_lock:
        if _segmenter is not None:
            _segmenter.unload()
            _segmenter = None

        if _classifier is not None:
            del _classifier
            _classifier = None

        _load_error = None

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Brain models unloaded, memory freed.")


# ── Validation ────────────────────────────────────────────────────────────

def _validate_upload(file: UploadFile, payload: bytes) -> Image.Image:
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type {file.content_type!r}; "
            f"expected one of {sorted(ALLOWED_CONTENT_TYPES)}.",
        )
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    if not payload.startswith(ALLOWED_MAGIC):
        raise HTTPException(
            status_code=400,
            detail="Upload is not a recognisable JPEG, PNG or WebP image.",
        )
    try:
        with Image.open(io.BytesIO(payload)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(payload)) as img:
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Image could not be decoded: {exc}"
        ) from exc


# ── Response models ───────────────────────────────────────────────────────

class SegmentationResult(BaseModel):
    """Full classification + segmentation response."""

    # Classification fields
    predicted_class: str
    confidence: float
    tumor_type: str
    sequence: str
    top5: list[dict[str, Any]]
    gradcam_base64: str
    gradcam_grid: list[int]
    explainability: dict[str, Any]
    locations_3d: list[dict[str, Any]]
    localization_basis: str
    model_trained: bool
    label_space_source: str
    inference_ms: float

    # Segmentation fields
    segmentation_performed: bool
    segmentation_skipped_reason: Optional[str] = None

    # Box-prompt segmentation (LiteMedSAM is a box-prompt-only model)
    box_prompt_used: bool = False
    box_mask_base64: Optional[str] = None
    box_overlay_base64: Optional[str] = None
    iou_box: Optional[float] = None
    box_coords: Optional[list[float]] = None

    # Metadata
    segmentation_input_size: Optional[str] = None
    original_size: Optional[str] = None
    total_ms: Optional[float] = None


class SegHealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    classifier_loaded: bool
    segmenter_loaded: bool
    segmentation_checkpoint_present: bool
    classification_checkpoint_present: bool
    detail: Optional[str] = None


class SegModelInfoResponse(BaseModel):
    segmentation: dict[str, Any]
    classification: dict[str, Any]


class LoadModelsResponse(BaseModel):
    status: str
    classifier_loaded: bool
    segmenter_loaded: bool
    load_time_ms: float


class UnloadModelsResponse(BaseModel):
    status: str
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.post("/segment", response_model=SegmentationResult)
async def segment_brain_tumor(file: UploadFile = File(...)) -> SegmentationResult:
    """Classify a brain MRI and, if a tumour is found, segment it.

    Pipeline:
    1. Run classifier with Grad-CAM.
    2. If predicted class is "Normal", return classification only.
    3. Otherwise, derive a bounding box from the Grad-CAM heatmap and use it
       as the prompt for LiteMedSAM segmentation. The released LiteMedSAM
       weights segment from boxes only, so no dense mask prompt is sent.
    """
    total_start = time.perf_counter()

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    image = _validate_upload(file, payload)

    classifier, segmenter = await _ensure_models_loaded()

    # ── Step 1: Classification with Grad-CAM ──────────────────────
    cls_start = time.perf_counter()
    try:
        cls_result = await run_in_threadpool(classifier.predict_with_gradcam, image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Classification inference failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Classification failed.") from exc

    cls_ms = (time.perf_counter() - cls_start) * 1000.0

    # 3D localisation
    locations = map_prediction_to_3d(
        cls_result["predicted_class"],
        locations=None,
        model_confidence=cls_result["confidence"],
    )

    base_response = {
        "predicted_class": cls_result["predicted_class"],
        "confidence": cls_result["confidence"],
        "tumor_type": cls_result["tumor_type"],
        "sequence": cls_result["sequence"],
        "top5": cls_result["top5"],
        "gradcam_base64": cls_result["gradcam_base64"],
        "gradcam_grid": cls_result["gradcam_grid"],
        "explainability": cls_result["explainability"],
        "locations_3d": locations,
        "localization_basis": LOCALIZATION_BASIS,
        "model_trained": cls_result["model_trained"],
        "label_space_source": cls_result["label_space_source"],
        "inference_ms": round(cls_ms, 1),
    }

    # ── Step 2: Check if tumour was detected ──────────────────────
    tumor_type = cls_result.get("tumor_type", "")
    is_normal = tumor_type.lower() == "normal" or cls_result["predicted_class"].lower().startswith("normal")

    if is_normal:
        logger.info("Classified as Normal — skipping segmentation.")
        return SegmentationResult(
            **base_response,
            segmentation_performed=False,
            segmentation_skipped_reason="Classified as Normal — no tumour region to segment.",
            total_ms=round((time.perf_counter() - total_start) * 1000.0, 1),
        )

    # ── Step 3: Segmentation with heatmap prompts ─────────────────

    # Reconstruct the Grad-CAM heatmap from the pipeline
    # The GradCAM.generate() returns a 2D normalised array —
    # we need to re-run it or extract from the classifier pipeline.
    # Since the classifier already ran predict_with_gradcam, we can
    # get the cam directly by running Grad-CAM again.
    from ..classification.model import GradCAM

    tensor = classifier.transform(image).unsqueeze(0).to(classifier.device)
    with GradCAM(classifier.model, classifier.target_layer) as grad_cam:
        cam = grad_cam.generate(tensor)

    seg_start = time.perf_counter()
    try:
        seg_result = await run_in_threadpool(
            segmenter.segment_with_heatmap, image, cam
        )
    except SegmentationUnavailableError as exc:
        logger.error("Segmentation failed: %s", exc)
        return SegmentationResult(
            **base_response,
            segmentation_performed=False,
            segmentation_skipped_reason=f"Segmentation error: {exc}",
            total_ms=round((time.perf_counter() - total_start) * 1000.0, 1),
        )
    except RuntimeError as exc:
        logger.error("Segmentation inference failed: %s", exc, exc_info=True)
        return SegmentationResult(
            **base_response,
            segmentation_performed=False,
            segmentation_skipped_reason=f"Segmentation inference error: {exc}",
            total_ms=round((time.perf_counter() - total_start) * 1000.0, 1),
        )

    seg_ms = (time.perf_counter() - seg_start) * 1000.0
    total_ms = (time.perf_counter() - total_start) * 1000.0

    logger.info(
        "Classification+Segmentation: class=%s conf=%.3f cls=%.0fms seg=%.0fms total=%.0fms",
        cls_result["predicted_class"],
        cls_result["confidence"],
        cls_ms,
        seg_ms,
        total_ms,
    )

    return SegmentationResult(
        **base_response,
        segmentation_performed=seg_result.get("segmentation_performed", True),
        box_prompt_used=seg_result.get("box_prompt_used", False),
        box_mask_base64=seg_result.get("box_mask_base64"),
        box_overlay_base64=seg_result.get("box_overlay_base64"),
        iou_box=seg_result.get("iou_box"),
        box_coords=seg_result.get("box_coords"),
        segmentation_input_size=seg_result.get("input_size"),
        original_size=seg_result.get("original_size"),
        total_ms=round(total_ms, 1),
    )


@router.post("/load-models", response_model=LoadModelsResponse)
async def load_brain_models() -> LoadModelsResponse:
    """Eagerly load both brain models into memory.

    Call this when the user enters the brain section.
    """
    started = time.perf_counter()
    try:
        classifier, segmenter = await _ensure_models_loaded()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return LoadModelsResponse(
        status="ok",
        classifier_loaded=classifier is not None,
        segmenter_loaded=segmenter is not None and segmenter.is_loaded,
        load_time_ms=round((time.perf_counter() - started) * 1000.0, 1),
    )


@router.post("/unload-models", response_model=UnloadModelsResponse)
async def unload_brain_models() -> UnloadModelsResponse:
    """Release both brain models from memory.

    Call this when the user leaves the brain section.
    """
    await _unload_models()
    return UnloadModelsResponse(
        status="ok",
        message="Both brain models unloaded. Memory freed.",
    )


@router.get("/seg-health", response_model=SegHealthResponse)
async def seg_health_check() -> SegHealthResponse:
    """Readiness probe for the segmentation pipeline."""
    cls_present = os.path.isfile(classification_checkpoint_path())
    seg_present = verify_weights()

    if _classifier is not None and _segmenter is not None:
        return SegHealthResponse(
            status="ok",
            classifier_loaded=True,
            segmenter_loaded=True,
            segmentation_checkpoint_present=seg_present,
            classification_checkpoint_present=cls_present,
        )

    detail = _load_error
    if not seg_present:
        detail = (
            "Segmentation checkpoint missing. Run: "
            "python -m app.organs.brain.segmentation.prepare_weights"
        )
    elif not cls_present:
        detail = "Classification checkpoint missing."

    return SegHealthResponse(
        status="ok" if (cls_present and seg_present) else "unavailable",
        classifier_loaded=_classifier is not None,
        segmenter_loaded=_segmenter is not None and _segmenter.is_loaded,
        segmentation_checkpoint_present=seg_present,
        classification_checkpoint_present=cls_present,
        detail=detail or "Checkpoints validated; models load on first request or /load-models.",
    )


@router.get("/seg-model-info", response_model=SegModelInfoResponse)
async def seg_model_info() -> SegModelInfoResponse:
    """Return metadata about both the classifier and segmenter."""
    seg_info: dict[str, Any] = {
        "model_name": "LiteMedSAM",
        "architecture": "TinyViT-256 + SAM PromptEncoder + MaskDecoder",
        "input_size": "256x256",
        "prompt_types": ["bounding_box"],
        "loaded": _segmenter is not None and _segmenter.is_loaded,
    }
    cls_info: dict[str, Any] = {
        "model_name": "EfficientNetV2-B2",
        "loaded": _classifier is not None,
        "num_classes": _classifier.num_classes if _classifier else len(CLASS_NAMES),
    }

    if _segmenter is not None and _segmenter.is_loaded:
        seg_info = _segmenter.model_info()
    if _classifier is not None:
        cls_info = _classifier.model_info()

    return SegModelInfoResponse(segmentation=seg_info, classification=cls_info)
