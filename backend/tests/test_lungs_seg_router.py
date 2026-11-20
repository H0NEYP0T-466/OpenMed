"""HTTP contract for the lungs segmentation router.

Five endpoints, and the rules that matter more than the happy path:

* a missing, unusable or empty checkpoint degrades the endpoint with a typed
  code — it never takes the process down;
* health answers from file presence and in-memory flags only, without building a
  network, so it can be polled cheaply;
* the response carries the scope notice and the three-value legend, because a
  mask without its legend is a picture, not a finding;
* a bad upload is rejected before any model work happens.

The segmenter is replaced by a stub that returns a real ``SegResult``; no
weights are opened and no network is built.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.organs.lungs.segmentation import router as seg
from app.organs.lungs.segmentation.checkpoint import SegmentationUnavailableError
from app.organs.lungs.segmentation.labels import MERGE_LUNG, MERGE_OPACITY, TASK_NAME
from app.organs.lungs.segmentation.pipeline import SegResult
from app.organs.lungs.segmentation.postprocess import LungPost, OpacityPost


# ─────────────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _png(size: int = 256) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((size, size), 128, dtype=np.uint8), mode="L").save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


def _result(opacity: bool = True) -> SegResult:
    lung = np.zeros((256, 256), dtype=np.uint8)
    lung[60:200, 40:110] = 1
    lung[60:200, 150:220] = 1
    opacity_mask = np.zeros((256, 256), dtype=np.uint8)
    opacity_mask[80:120, 150:200] = 1
    label_map = np.zeros((256, 256), dtype=np.uint8)
    label_map[lung > 0] = MERGE_LUNG
    if opacity:
        label_map[opacity_mask > 0] = MERGE_OPACITY
    return SegResult(
        lung_mask=lung,
        opacity_mask=opacity_mask if opacity else None,
        lung_probability=lung.astype(np.float32),
        opacity_probability=opacity_mask.astype(np.float32) if opacity else None,
        label_map=label_map,
        original_size=(256, 256),
        crop_box=[40, 60, 220, 200] if opacity else None,
        crop_fallback=False,
        opacity_available=opacity,
        opacity_unavailable_reason=None if opacity else "checkpoint has no opacity head",
        lung_post=LungPost(),
        opacity_post=OpacityPost() if opacity else None,
        inference_ms=12.5,
        stats={
            "lung_area_px": int(lung.sum()),
            "lung_fraction_of_frame": float(lung.sum()) / lung.size,
            "lung_components": 2,
            "lung_bbox": [40, 60, 220, 200],
            "opacity_area_px": int(opacity_mask.sum()) if opacity else None,
            "opacity_fraction_of_lung": (float(opacity_mask.sum()) / float(lung.sum())
                                         if opacity else None),
            "opacity_components": 1 if opacity else None,
            "opacity_peak_probability": 0.81 if opacity else None,
            "opacity_mean_probability": 0.62 if opacity else None,
            "crop_box": [40, 60, 220, 200] if opacity else None,
            "crop_fallback": False,
        },
    )


class _StubSegmenter:
    """Stands in for :class:`LungSegmenter` with the attributes the router reads."""

    def __init__(self, opacity: bool = True) -> None:
        self.opacity = opacity
        self.is_loaded = True
        self.device = "cpu"
        self.unloaded = False

    def predict_gray(self, gray: np.ndarray, **_kwargs) -> SegResult:
        return _result(self.opacity)

    def unload(self) -> None:
        self.unloaded = True
        self.is_loaded = False

    def model_info(self) -> dict:
        return {"checkpoint": "lungs_seg.pth", "loaded": self.is_loaded,
                "task": TASK_NAME, "heads": ["lung", "opacity"]}


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setattr(seg, "segmentation_checkpoint_path",
                        lambda: str(tmp_path / "absent.pth"))
    monkeypatch.setattr(seg, "_segmenter", None)
    monkeypatch.setattr(seg, "_segmenter_error", None)
    monkeypatch.setattr(seg, "_segmenter_error_key", None)
    app = FastAPI()
    app.include_router(seg.router, prefix="/api/lungs")
    return TestClient(app)


@pytest.fixture()
def loaded_client(client: TestClient, monkeypatch) -> TestClient:
    monkeypatch.setattr(seg, "_segmenter", _StubSegmenter(True))
    return client


# ─────────────────────────────────────────────────────────────────────────────
# health
# ─────────────────────────────────────────────────────────────────────────────


def test_health_reports_a_missing_checkpoint_without_failing(client: TestClient) -> None:
    response = client.get("/api/lungs/seg-health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "unavailable"
    assert body["checkpoint_present"] is False
    assert body["model_loaded"] is False
    assert "lungs_seg.pth" in body["detail"]


def test_health_reports_residency_when_loaded(loaded_client: TestClient) -> None:
    body = loaded_client.get("/api/lungs/seg-health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


# ─────────────────────────────────────────────────────────────────────────────
# segmentation
# ─────────────────────────────────────────────────────────────────────────────


def test_segment_refuses_with_a_typed_code_when_the_checkpoint_is_absent(
    client: TestClient,
) -> None:
    response = client.post("/api/lungs/segment",
                           files={"file": ("scan.png", _png(), "image/png")})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "checkpoint_missing"
    assert "lungs_seg.pth" in detail["message"]


def test_segment_returns_masks_stats_and_the_three_value_legend(
    loaded_client: TestClient,
) -> None:
    response = loaded_client.post("/api/lungs/segment",
                                  files={"file": ("scan.png", _png(), "image/png")})
    assert response.status_code == 200

    body = response.json()
    assert body["task"] == TASK_NAME
    assert body["opacity_available"] is True
    assert body["image_size"] == {"width": 256, "height": 256}
    assert body["mask_size"] == {"width": 256, "height": 256}

    # masks travel as PNG data URLs, not as json arrays
    assert body["lung_mask_png"].startswith("data:image/png;base64,")
    assert body["opacity_mask_png"].startswith("data:image/png;base64,")
    assert body["label_map_png"].startswith("data:image/png;base64,")

    # the legend is the merge rule, spelled out for a reader who never saw the code
    assert [item["value"] for item in body["legend"]] == [0, 1, 2]
    assert "opacity" in body["legend"][2]["name"].lower()

    assert body["stats"]["lung_area_px"] > 0
    assert body["stats"]["opacity_area_px"] > 0
    assert body["inference_ms"] > 0
    # the scope notice travels with every response
    assert "not a diagnosis" in body["scope_notice"]
    assert body["postprocess"]["lung"]["threshold"] == 0.5


def test_segment_without_an_opacity_head_says_why(loaded_client: TestClient,
                                                  monkeypatch) -> None:
    monkeypatch.setattr(seg, "_segmenter", _StubSegmenter(False))
    body = loaded_client.post("/api/lungs/segment",
                              files={"file": ("scan.png", _png(), "image/png")}).json()

    assert body["opacity_available"] is False
    assert body["opacity_mask_png"] is None
    assert "opacity head" in body["opacity_unavailable_reason"]


def test_segment_rejects_a_wrong_content_type(loaded_client: TestClient) -> None:
    response = loaded_client.post("/api/lungs/segment",
                                  files={"file": ("scan.txt", b"hello", "text/plain")})
    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "unsupported_media"


def test_segment_rejects_a_file_whose_bytes_are_not_an_image(loaded_client: TestClient) -> None:
    response = loaded_client.post("/api/lungs/segment",
                                  files={"file": ("scan.png", b"not an image", "image/png")})
    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "unsupported_media"


def test_inference_failure_is_a_500_not_a_crash(loaded_client: TestClient,
                                                monkeypatch) -> None:
    class _Broken(_StubSegmenter):
        def predict_gray(self, gray, **_kwargs):
            raise RuntimeError("cuda device-side assert")

    monkeypatch.setattr(seg, "_segmenter", _Broken(True))
    response = loaded_client.post("/api/lungs/segment",
                                  files={"file": ("scan.png", _png(), "image/png")})
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "inference_failed"


# ─────────────────────────────────────────────────────────────────────────────
# residency lifecycle
# ─────────────────────────────────────────────────────────────────────────────


def test_warmup_answers_200_even_when_nothing_can_load(client: TestClient) -> None:
    response = client.post("/api/lungs/seg-warmup")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["checkpoint_present"] is False


def test_unload_reports_whether_anything_was_resident(client: TestClient,
                                                      monkeypatch) -> None:
    assert client.post("/api/lungs/seg-unload").json()["unloaded"] is False

    stub = _StubSegmenter()
    monkeypatch.setattr(seg, "_segmenter", stub)
    body = client.post("/api/lungs/seg-unload").json()

    assert body["unloaded"] is True
    assert stub.unloaded is True
    assert seg._segmenter is None       # the resident slot is actually cleared


def test_model_info_degrades_without_a_checkpoint(client: TestClient) -> None:
    body = client.get("/api/lungs/seg-model-info").json()
    assert body["available"] is False
    assert "lungs_seg.pth" in body["detail"]


def test_model_info_reports_the_loaded_model(loaded_client: TestClient) -> None:
    body = loaded_client.get("/api/lungs/seg-model-info").json()
    assert body["available"] is True
    assert body["info"]["task"] == TASK_NAME


def test_router_surfaces_the_shared_error_type() -> None:
    """The pipeline raises one typed error; the router maps it to 503 everywhere."""
    error = SegmentationUnavailableError("nope", code="checkpoint_unusable")
    assert error.code == "checkpoint_unusable"
