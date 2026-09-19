"""FastAPI router for brain tumour classification.

Endpoints
---------
POST /api/brain/classify    Upload an MRI image -> classification + Grad-CAM
GET  /api/brain/model-info   Model metadata derived from the loaded label space
GET  /api/brain/health       Readiness probe reflecting the real model state
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from typing import Any, Literal, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .brain_regions import LOCALIZATION_BASIS, map_prediction_to_3d
from .label_space import (
    CLASS_NAMES,
    LabelSpaceError,
    label_space_path_for,
    read_label_space,
)
from .model import MODEL_TAG
from .pipeline import BrainClassificationPipeline

logger = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_CHECKPOINT = "brain_best_model.pth"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"RIFF")

_pipeline: Optional[BrainClassificationPipeline] = None
_pipeline_error: Optional[str] = None
_pipeline_error_key: Optional[tuple[float, int]] = None
_load_lock = asyncio.Lock()

_dataset_summary_cache: Optional[dict[str, Any]] = None


def checkpoint_path() -> str:
    configured = os.getenv("OPENMED_BRAIN_CHECKPOINT")
    if configured:
        return os.path.expanduser(configured)
    return os.path.join(os.path.dirname(__file__), "checkpoints", DEFAULT_CHECKPOINT)


def _dataset_summary() -> dict[str, Any]:
    global _dataset_summary_cache
    if _dataset_summary_cache is not None:
        return _dataset_summary_cache

    summary: dict[str, Any] = {"samples": None, "tumour_types": None, "sequences": []}
    data_root = os.getenv("OPENMED_BRAIN_DATASET")
    if not data_root:
        _dataset_summary_cache = summary
        return summary

    try:
        from .dataset import BrainTumorDataset

        json_path, _ = BrainTumorDataset.locate_data_and_images(data_root)

        with open(json_path) as handle:
            raw = json.load(handle)
        rows = {
            key: value
            for key, value in raw.items()
            if not key.endswith(BrainTumorDataset.MASK_SUFFIX)
        }
        summary = {
            "samples": len(rows),
            "tumour_types": len({v["tumor_type"] for v in rows.values()}),
            "sequences": sorted({v.get("sequence", "") for v in rows.values()}),
        }
    except Exception as exc:  # noqa: BLE001 - metadata is best effort
        logger.warning("Could not summarise dataset metadata: %s", exc)

    _dataset_summary_cache = summary
    return summary


def _declared_label_space():
    """Read the checkpoint's sidecar label file without loading any weights."""
    path = checkpoint_path()
    try:
        return read_label_space(path)
    except LabelSpaceError as exc:
        logger.error("Unreadable label file: %s", exc)
        return None


def _checkpoint_identity(path: str) -> Optional[tuple[float, int]]:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (stat.st_mtime, stat.st_size)


def preflight() -> tuple[bool, Optional[str], list[str]]:
    """Validate the configured checkpoint cheaply, without loading the network.

    Returns (usable, blocking_reason, warnings). Only structurally impossible
    states block: the hard guarantee that head width and class names agree
    belongs to `resolve_label_space`, which also accepts a correctly-paired
    non-canonical model.
    """
    path = checkpoint_path()
    warnings: list[str] = []

    if not os.path.isfile(path):
        return False, f"Brain classifier checkpoint not found at {path}.", warnings

    if os.path.getsize(path) == 0:
        return False, f"Brain classifier checkpoint at {path} is empty.", warnings

    space = _declared_label_space()
    if space is None:
        warnings.append(
            f"No label file beside {os.path.basename(path)}; class order is being "
            "assumed from the canonical label space. Copy the .label_space.json "
            "produced by training next to the checkpoint to verify it."
        )
    elif space.num_classes != len(CLASS_NAMES):
        warnings.append(
            f"{os.path.basename(label_space_path_for(path))} declares "
            f"{space.num_classes} classes, differing from the {len(CLASS_NAMES)}-class "
            "canonical brain label space. Serving binds names from the sidecar."
        )

    return True, None, warnings


async def _get_pipeline() -> BrainClassificationPipeline:
    """Lazily build the singleton, failing closed with a recorded reason."""
    global _pipeline, _pipeline_error, _pipeline_error_key

    if _pipeline is not None:
        return _pipeline

    path = checkpoint_path()
    identity = _checkpoint_identity(path)

    if _pipeline_error is not None and _pipeline_error_key == identity:
        raise HTTPException(
            status_code=503,
            detail=f"Brain classifier unavailable: {_pipeline_error}",
        )
    if _pipeline_error is not None and _pipeline_error_key != identity:
        # The checkpoint was replaced or repaired since the failed attempt.
        logger.info("Checkpoint changed since the last failure; retrying load.")
        _pipeline_error = None
        _pipeline_error_key = None

    async with _load_lock:
        if _pipeline is not None:
            return _pipeline

        started = time.perf_counter()
        logger.info("Loading brain classification model from %s", path)
        try:
            _pipeline = await run_in_threadpool(
                lambda: BrainClassificationPipeline(model_path=path)
            )
        except Exception as exc:  # noqa: BLE001 - any load failure is fail-closed
            _pipeline_error = str(exc) or type(exc).__name__
            _pipeline_error_key = identity
            logger.error("Brain model unavailable: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=503, detail=f"Brain classifier unavailable: {_pipeline_error}"
            ) from exc

        logger.info(
            "Brain model ready in %.2fs (classes=%d, trained=%s)",
            time.perf_counter() - started,
            _pipeline.num_classes,
            _pipeline.using_trained_weights,
        )
        return _pipeline


class Top5Prediction(BaseModel):
    model_config = {"populate_by_name": True}

    class_name: str = Field(alias="class", serialization_alias="class")
    confidence: float


class BrainRegion(BaseModel):
    name: str
    display_name: str
    coordinates_3d: list[float]
    color: str
    lobe: str
    description: str
    probability: float
    rank: int
    basis: str


class ClassificationResponse(BaseModel):
    predicted_class: str
    confidence: float
    tumor_type: str
    sequence: str
    top5: list[Top5Prediction]
    probabilities: list[float]
    gradcam_base64: str
    gradcam_grid: list[int]
    explainability: dict[str, Any]
    locations_3d: list[BrainRegion]
    localization_basis: str
    model_trained: bool
    label_space_source: str
    inference_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    trained_weights: bool
    checkpoint_present: bool
    detail: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


class ModelInfoResponse(BaseModel):
    model_name: str
    model_tag: str
    declared_model_tag: str
    architecture_matches_declaration: bool
    num_classes: int
    class_names: list[str]
    input_size: str
    task: str
    dataset: dict[str, Any]
    label_space_source: str
    trained_weights_loaded: bool
    metrics: dict[str, Any]


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
            status_code=400, detail="Upload is not a recognisable JPEG, PNG or WebP image."
        )

    try:
        with Image.open(io.BytesIO(payload)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Image could not be decoded: {exc}"
        ) from exc


@router.post("/classify", response_model=ClassificationResponse)
async def classify_brain_tumor(file: UploadFile = File(...)) -> ClassificationResponse:
    """Classify a brain MRI image and return Grad-CAM and localisation hints."""
    started = time.perf_counter()
    logger.info("Inference request file=%s content_type=%s", file.filename, file.content_type)

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    image = _validate_upload(file, payload)
    pipeline = await _get_pipeline()

    try:
        result = await run_in_threadpool(pipeline.predict_with_gradcam, image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Inference failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail="Inference failed while processing the image."
        ) from exc

    locations = map_prediction_to_3d(
        result["predicted_class"],
        locations=None,
        model_confidence=result["confidence"],
    )
    result["locations_3d"] = locations
    result["localization_basis"] = LOCALIZATION_BASIS
    result["inference_ms"] = round((time.perf_counter() - started) * 1000.0, 1)

    logger.info(
        "Prediction class=%s conf=%.3f tumor_type=%s in %.0fms",
        result["predicted_class"],
        result["confidence"],
        result["tumor_type"],
        result["inference_ms"],
    )
    return ClassificationResponse(**result)


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info() -> ModelInfoResponse:
    """Return model metadata derived from the live label space, not hardcoded copy."""
    summary = _dataset_summary()
    if _pipeline is not None:
        info = _pipeline.model_info()
    else:
        # Report what the checkpoint declares without loading weights, so the UI
        # can show the real class count and metrics before the first inference.
        declared = _declared_label_space()
        info = {
            "model_name": "EfficientNetV2-B2",
            "model_tag": (declared.model_tag if declared else "") or MODEL_TAG,
            "declared_model_tag": (declared.model_tag if declared else "") or MODEL_TAG,
            "architecture_matches_declaration": True,
            "num_classes": declared.num_classes if declared else len(CLASS_NAMES),
            "class_names": list(declared.class_names) if declared else list(CLASS_NAMES),
            "input_size": "resolved on first inference",
            "label_space_source": declared.source if declared else "canonical (unverified)",
            "trained_weights_loaded": False,
            "metrics": declared.metrics if declared else {},
        }

    return ModelInfoResponse(
        model_name=info["model_name"],
        model_tag=info["model_tag"],
        declared_model_tag=info["declared_model_tag"],
        architecture_matches_declaration=info["architecture_matches_declaration"],
        num_classes=info["num_classes"],
        class_names=info["class_names"],
        input_size=info["input_size"],
        task=f"{info['num_classes']}-class brain tumour and sequence classification",
        dataset=summary,
        label_space_source=info["label_space_source"],
        trained_weights_loaded=info["trained_weights_loaded"],
        metrics=info["metrics"],
    )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Readiness probe reporting load state, and the reason when not ready."""
    path = checkpoint_path()
    present = os.path.isfile(path)

    if _pipeline is not None:
        return HealthResponse(
            status="ok",
            model_loaded=True,
            trained_weights=_pipeline.using_trained_weights,
            checkpoint_present=True,
        )

    usable, reason, warnings = preflight()
    if _pipeline_error is not None:
        usable, reason = False, _pipeline_error

    return HealthResponse(
        status="ok" if usable else "unavailable",
        model_loaded=False,
        trained_weights=False,
        checkpoint_present=present,
        detail=_pipeline_error
        or reason
        or "Checkpoint validated; the network loads on the first request.",
        warnings=warnings,
    )
