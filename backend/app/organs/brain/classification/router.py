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
from .label_space import CLASS_NAMES
from .pipeline import BrainClassificationPipeline, ModelUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_CHECKPOINT = "brain_best_model.pth"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"RIFF")

_pipeline: Optional[BrainClassificationPipeline] = None
_pipeline_error: Optional[str] = None
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


async def _get_pipeline() -> BrainClassificationPipeline:
    """Lazily build the singleton, failing closed with a recorded reason."""
    global _pipeline, _pipeline_error

    if _pipeline is not None:
        return _pipeline
    if _pipeline_error is not None:
        raise HTTPException(
            status_code=503,
            detail=f"Brain classifier unavailable: {_pipeline_error}",
        )

    async with _load_lock:
        if _pipeline is not None:
            return _pipeline

        path = checkpoint_path()
        started = time.perf_counter()
        logger.info("Loading brain classification model from %s", path)
        try:
            _pipeline = await run_in_threadpool(
                lambda: BrainClassificationPipeline(model_path=path)
            )
        except ModelUnavailableError as exc:
            _pipeline_error = str(exc)
            logger.error("Brain model unavailable: %s", exc)
            raise HTTPException(status_code=503, detail=f"{exc}") from exc

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
    detail: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_tag: str
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
        info = {
            "model_name": "EfficientNetV2-B2",
            "model_tag": "tf_efficientnetv2_b2.in1k",
            "num_classes": len(CLASS_NAMES),
            "class_names": list(CLASS_NAMES),
            "input_size": "pending first inference",
            "label_space_source": "canonical",
            "trained_weights_loaded": False,
            "metrics": {},
        }

    return ModelInfoResponse(
        model_name=info["model_name"],
        model_tag=info["model_tag"],
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
    """Readiness probe reporting load state and any recorded failure reason."""
    if _pipeline is not None:
        return HealthResponse(
            status="ok",
            model_loaded=True,
            trained_weights=_pipeline.using_trained_weights,
        )
    return HealthResponse(
        status="unavailable",
        model_loaded=False,
        trained_weights=False,
        detail=_pipeline_error or "Model has not been loaded yet.",
    )
