"""HTTP contract tests for the brain classification router.

Every test drives the real FastAPI app through its test client against a
synthetic checkpoint, so the label-space guard and the fail-closed behaviour are
covered, not just the happy path.
"""

from __future__ import annotations

import io
import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.organs.brain.classification import router as router_module
from app.organs.brain.classification.label_space import CLASS_NAMES, save_label_space


def _jpeg_bytes(seed: int = 0, size: int = 300) -> bytes:
    array = np.random.default_rng(seed).integers(0, 255, size=(size, size), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).convert("L").resize((size, size)).convert("RGB").save(
        buffer, "JPEG"
    )
    return buffer.getvalue()


@pytest.fixture()
def reset_singleton(monkeypatch):
    monkeypatch.setattr(router_module, "_pipeline", None, raising=False)
    monkeypatch.setattr(router_module, "_pipeline_error", None, raising=False)
    yield


@pytest.fixture()
def trained_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from app.organs.brain.classification.model import create_model

    path = tmp_path / "brain_best_model.pth"
    torch.save(create_model(num_classes=len(CLASS_NAMES), pretrained=False).state_dict(), path)
    save_label_space(
        str(path), CLASS_NAMES, model_tag="unit-test-backbone",
        metrics={"test_acc": 0.42, "monitor": "val_loss"},
    )
    return str(path)


@pytest.fixture()
def client(reset_singleton, monkeypatch):
    from app.main import app

    return TestClient(app)


def test_health_is_honest_when_checkpoint_missing(
    reset_singleton, monkeypatch, client
) -> None:
    monkeypatch.setenv("OPENMED_BRAIN_CHECKPOINT", "/nonexistent/absent.pth")

    health = client.get("/api/brain/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "unavailable"
    assert body["model_loaded"] is False
    assert "not found" in body["detail"]

    response = client.post(
        "/api/brain/classify",
        files={"file": ("scan.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 503
    assert "not found" in response.json()["detail"]


def test_legacy_head_width_is_refused_not_misread(
    reset_singleton, monkeypatch, client, tmp_path
) -> None:
    torch = pytest.importorskip("torch")
    from app.organs.brain.classification.model import create_model

    path = tmp_path / "legacy.pth"
    torch.save(create_model(num_classes=39, pretrained=False).state_dict(), path)
    monkeypatch.setenv("OPENMED_BRAIN_CHECKPOINT", str(path))

    response = client.post(
        "/api/brain/classify",
        files={"file": ("scan.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "39 logits" in detail and "42 names" in detail


def test_model_info_reports_declared_label_space_before_loading(
    reset_singleton, monkeypatch, client, trained_checkpoint
) -> None:
    monkeypatch.setenv("OPENMED_BRAIN_CHECKPOINT", trained_checkpoint)

    info = client.get("/api/brain/model-info").json()
    assert info["num_classes"] == len(CLASS_NAMES)
    assert info["class_names"] == list(CLASS_NAMES)
    assert "42-class" in info["task"]
    assert info["label_space_source"].startswith("sidecar:")
    assert info["metrics"]["test_acc"] == 0.42
    assert info["trained_weights_loaded"] is False
    assert info["input_size"] == "resolved on first inference"


def test_model_info_reports_resolved_config_after_inference(
    reset_singleton, monkeypatch, client, trained_checkpoint
) -> None:
    monkeypatch.setenv("OPENMED_BRAIN_CHECKPOINT", trained_checkpoint)
    client.post(
        "/api/brain/classify",
        files={"file": ("scan.jpg", _jpeg_bytes(4), "image/jpeg")},
    )

    info = client.get("/api/brain/model-info").json()
    assert info["trained_weights_loaded"] is True
    assert info["input_size"] == "208x208"
    # The sidecar names an architecture timm has never heard of, so the pipeline
    # falls back to the project default rather than crashing.
    assert info["declared_model_tag"] == "unit-test-backbone"
    assert info["architecture_matches_declaration"] is False


def test_model_info_works_before_any_inference(reset_singleton, client) -> None:
    info = client.get("/api/brain/model-info").json()
    assert info["num_classes"] == len(CLASS_NAMES)
    assert info["trained_weights_loaded"] is False


def test_classify_returns_the_full_contract(
    reset_singleton, monkeypatch, client, trained_checkpoint
) -> None:
    monkeypatch.setenv("OPENMED_BRAIN_CHECKPOINT", trained_checkpoint)

    response = client.post(
        "/api/brain/classify",
        files={"file": ("scan.jpg", _jpeg_bytes(1), "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["predicted_class"] in CLASS_NAMES
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top5"]) == 5
    assert {row["class"] for row in body["top5"]} <= set(CLASS_NAMES)
    assert len(body["probabilities"]) == len(CLASS_NAMES)
    assert body["gradcam_base64"].startswith("data:image/jpeg;base64,")
    assert body["gradcam_grid"] == [7, 7]
    assert body["explainability"]["method"] == "gradcam"
    assert body["localization_basis"]
    assert body["model_trained"] is True
    assert "simulated" not in body

    for region in body["locations_3d"]:
        assert region["basis"] in {"tumour_type_prior", "image_metadata"}
        assert any(abs(value) > 5 for value in region["coordinates_3d"])


def test_classify_rejects_non_image_payloads(reset_singleton, client) -> None:
    wrong_type = client.post(
        "/api/brain/classify",
        files={"file": ("scan.txt", b"hello world", "text/plain")},
    )
    assert wrong_type.status_code == 415

    mislabelled = client.post(
        "/api/brain/classify",
        files={"file": ("scan.jpg", b"%PDF-1.4 not an image", "image/jpeg")},
    )
    assert mislabelled.status_code == 400


def test_classify_enforces_upload_size_limit(reset_singleton, client, monkeypatch) -> None:
    monkeypatch.setattr(router_module, "MAX_UPLOAD_BYTES", 1024)
    response = client.post(
        "/api/brain/classify",
        files={"file": ("big.jpg", _jpeg_bytes(2, size=400), "image/jpeg")},
    )
    assert response.status_code == 413


def test_dataset_summary_is_absent_without_env(reset_singleton, monkeypatch, client) -> None:
    monkeypatch.delenv("OPENMED_BRAIN_DATASET", raising=False)
    monkeypatch.setattr(router_module, "_dataset_summary_cache", None, raising=False)
    body = client.get("/api/brain/model-info").json()
    assert body["dataset"]["samples"] is None


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "datasets", "brain", "archive", "DATA.json")),
    reason="brain dataset not present",
)
def test_dataset_summary_counts_the_real_manifest(
    reset_singleton, monkeypatch, client
) -> None:
    archive = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datasets", "brain", "archive",
    )
    monkeypatch.setenv("OPENMED_BRAIN_DATASET", archive)
    monkeypatch.setattr(router_module, "_dataset_summary_cache", None, raising=False)

    dataset = client.get("/api/brain/model-info").json()["dataset"]
    with open(os.path.join(archive, "DATA.json")) as handle:
        raw = json.load(handle)
    expected = len([k for k in raw if not k.endswith("_mask.png")])

    assert dataset["samples"] == expected
    assert dataset["tumour_types"] == 14
    assert set(dataset["sequences"]) == {"T1", "T1C+", "T2"}
