"""HTTP contract tests for the /api/brain/segment prompt-mode toggle.

The toggle decides what LiteMedSAM is prompted with: the bounding box derived
from the Grad-CAM heatmap, or nothing at all. These tests drive the real FastAPI
app against stub pipelines so the wiring is covered without loading either real
checkpoint — and so that each mode is shown to reach the *right* pipeline
method, rather than merely returning a 200.

The Normal-skips-segmentation rule is asserted in both modes, because it is the
one behaviour the toggle must never change.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.organs.brain.segmentation import router as seg_router


def _jpeg_bytes(seed: int = 0, size: int = 288) -> bytes:
    array = np.random.default_rng(seed).integers(0, 255, size=(size, size), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).convert("L").convert("RGB").save(buffer, "JPEG")
    return buffer.getvalue()


class _StubClassifier:
    """Just enough classifier to satisfy the endpoint's base_response."""

    device = "cpu"
    target_layer = None

    def __init__(self, predicted_class: str = "Gliomas", confidence: float = 0.87) -> None:
        self._predicted_class = predicted_class
        self._confidence = confidence

    def predict_with_gradcam(self, image):
        return {
            "predicted_class": self._predicted_class,
            "confidence": self._confidence,
            "tumor_type": self._predicted_class,
            "sequence": "T1C+",
            "top5": [{"class": self._predicted_class, "confidence": self._confidence}],
            "gradcam_base64": "data:image/jpeg;base64,AAAA",
            "gradcam_grid": [7, 7],
            # The router derives the box from this map rather than recomputing
            # one, so the stub has to supply it.
            "cam_array": np.zeros((7, 7), dtype=np.float32),
            "explainability": {
                "method": "gradcam++",
                "layer": "stub",
                "interpretation": "stub",
            },
            "model_trained": True,
            "label_space_source": "unit-test",
        }


class _StubSegmenter:
    """Records which pipeline method ran, so the mode routing can be asserted."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.heatmap_cams: list[object] = []
        self.clicks: list[tuple[float, float, int]] = []

    def segment_with_heatmap(self, image, cam):
        self.calls.append("segment_with_heatmap")
        self.heatmap_cams.append(cam)
        return {
            "segmentation_performed": True,
            "prompt_mode": "heatmap_box",
            "box_prompt_used": True,
            "box_mask_base64": "data:image/png;base64,SEGHEAT",
            "box_overlay_base64": "data:image/jpeg;base64,OVLHEAT",
            "iou_box": 0.91,
            "mask_foreground_px": 14956,
            "box_coords": [10.0, 20.0, 200.0, 210.0],
            "point_prompt_used": True,
            "point_coords": [150.0, 112.0],
            "input_size": "256x256",
            "original_size": "288x288",
        }

    def segment_raw(self, image):
        self.calls.append("segment_raw")
        return {
            "segmentation_performed": True,
            "prompt_mode": "raw",
            "seg_mask_base64": "data:image/png;base64,SEGRAW",
            "seg_overlay_base64": "data:image/jpeg;base64,OVLRAW",
            "iou_pred": 0.42,
            # The real weights return an empty mask when unprompted.
            "mask_foreground_px": 0,
            "input_size": "256x256",
            "original_size": "288x288",
        }

    def segment_with_click(self, image, click_x, click_y, box_size=48):
        self.calls.append("segment_with_click")
        self.clicks.append((click_x, click_y, box_size))
        return {
            "segmentation_performed": True,
            "prompt_mode": "click_box",
            "seg_mask_base64": "data:image/png;base64,SEGCLICK",
            "seg_overlay_base64": "data:image/jpeg;base64,OVLCLICK",
            "iou_pred": 0.598,
            "mask_foreground_px": 3390,
            "click": [click_x, click_y],
            "box_coords": [
                click_x - 24, click_y - 24, click_x + 24, click_y + 24,
            ],
            "input_size": "256x256",
            "original_size": "288x288",
        }


def _ensure_factory(classifier, segmenter):
    """Build a stand-in for the router's async model loader."""

    async def _fake_ensure():
        return classifier, segmenter

    return _fake_ensure


@pytest.fixture()
def wired(monkeypatch):
    """Patch the router's model loading with stubs.

    Yields the stub segmenter so tests can assert which method was reached.
    """
    pytest.importorskip("torch")

    segmenter = _StubSegmenter()
    classifier = _StubClassifier()

    monkeypatch.setattr(
        seg_router, "_ensure_models_loaded", _ensure_factory(classifier, segmenter)
    )

    from app.main import app

    return TestClient(app), segmenter


def test_segment_defaults_to_the_heatmap_prompt(wired) -> None:
    """Omitting the flag must keep the original, heatmap-prompted behaviour."""
    client, segmenter = wired

    response = client.post(
        "/api/brain/segment",
        files={"file": ("scan.jpg", _jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert segmenter.calls == ["segment_with_heatmap"]
    assert body["prompt_mode"] == "heatmap_box"
    assert body["seg_mask_base64"] == "data:image/png;base64,SEGHEAT"
    assert body["iou_pred"] == 0.91
    assert body["mask_foreground_px"] == 14956
    assert body["box_coords"] == [10.0, 20.0, 200.0, 210.0]
    assert body["point_prompt_used"] is True
    assert body["point_coords"] == [150.0, 112.0]
    # The box must come from the classifier's own map, not a recomputed one.
    assert len(segmenter.heatmap_cams) == 1
    assert np.array_equal(segmenter.heatmap_cams[0], np.zeros((7, 7), dtype=np.float32))


def test_segment_raw_mode_never_builds_a_heatmap_prompt(wired) -> None:
    """With the flag cleared, no prompt is derived and no CAM is computed."""
    client, segmenter = wired

    response = client.post(
        "/api/brain/segment",
        data={"use_heatmap_prompt": "false"},
        files={"file": ("scan.jpg", _jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert segmenter.calls == ["segment_raw"]
    assert body["prompt_mode"] == "raw"
    assert body["seg_mask_base64"] == "data:image/png;base64,SEGRAW"
    assert body["iou_pred"] == 0.42
    # An empty mask must survive the API as 0, not as None. None already means
    # "no mask was produced"; collapsing the two would make the UI unable to
    # distinguish a model that found nothing from one that never ran.
    assert body["mask_foreground_px"] == 0
    # Raw mode is not a box prompt: the box fields must stay empty, and the
    # activation map must never have been handed to the segmenter.
    assert body["box_prompt_used"] is False
    assert body["box_coords"] is None
    assert body["point_prompt_used"] is False
    assert body["point_coords"] is None
    assert segmenter.heatmap_cams == []


@pytest.mark.parametrize("flag", ["true", "false"])
def test_normal_prediction_skips_segmentation_in_both_modes(
    wired, monkeypatch, flag: str
) -> None:
    """The Normal short-circuit is independent of the prompt mode."""
    client, segmenter = wired
    monkeypatch.setattr(
        seg_router,
        "_ensure_models_loaded",
        _ensure_factory(
            _StubClassifier(predicted_class="Normal", confidence=0.99), segmenter
        ),
    )

    response = client.post(
        "/api/brain/segment",
        data={"use_heatmap_prompt": flag},
        files={"file": ("scan.jpg", _jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert segmenter.calls == []  # the segmenter was never touched
    assert body["segmentation_performed"] is False
    assert "Normal" in body["segmentation_skipped_reason"]
    assert body["prompt_mode"] == ("heatmap_box" if flag == "true" else "raw")


@pytest.fixture()
def wired_click(monkeypatch):
    """Patch the router's segmenter-only loader for the click path."""
    pytest.importorskip("torch")

    segmenter = _StubSegmenter()

    async def _fake_seg():
        return segmenter

    monkeypatch.setattr(seg_router, "_ensure_segmenter_loaded", _fake_seg)

    from app.main import app

    return TestClient(app), segmenter


def test_click_segment_builds_a_box_around_the_click(wired_click) -> None:
    """The click is echoed in 256² space and forwarded with the box size."""
    client, segmenter = wired_click

    response = client.post(
        "/api/brain/segment-click",
        data={"click_x": 0.6, "click_y": 0.45, "box_size": 48},
        files={"file": ("scan.jpg", _jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert segmenter.calls == ["segment_with_click"]
    # Normalised click -> 256×256 prompt space.
    assert segmenter.clicks == [(0.6 * 255.0, 0.45 * 255.0, 48)]
    assert body["prompt_mode"] == "click_box"
    assert body["click"] == [0.6 * 255.0, 0.45 * 255.0]
    assert body["seg_mask_base64"] == "data:image/png;base64,SEGCLICK"
    assert body["mask_foreground_px"] == 3390


def test_click_segment_never_loads_the_classifier(wired_click, monkeypatch) -> None:
    """The click path must not pay for a classification pass.

    Patched to explode rather than to return stubs: if the endpoint reached for
    the classifier, this test fails instead of silently passing.
    """
    client, _ = wired_click

    def _boom():
        raise AssertionError("click path must not load the classifier")

    monkeypatch.setattr(seg_router, "_ensure_models_loaded", _boom)

    response = client.post(
        "/api/brain/segment-click",
        data={"click_x": 0.6, "click_y": 0.45},
        files={"file": ("scan.jpg", _jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200


def test_click_segment_rejects_out_of_range_coordinates(wired_click) -> None:
    """Coordinates are normalised 0–1; anything else is a client error."""
    client, segmenter = wired_click

    for cx, cy in (("1.5", "0.4"), ("0.4", "-0.1")):
        response = client.post(
            "/api/brain/segment-click",
            data={"click_x": cx, "click_y": cy},
            files={"file": ("scan.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 422, f"({cx},{cy}) should be refused"

    assert segmenter.calls == []


def test_click_segment_rejects_a_bad_box_size(wired_click) -> None:
    client, segmenter = wired_click

    response = client.post(
        "/api/brain/segment-click",
        data={"click_x": 0.6, "click_y": 0.45, "box_size": 8},
        files={"file": ("scan.jpg", _jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 422
    assert segmenter.calls == []
