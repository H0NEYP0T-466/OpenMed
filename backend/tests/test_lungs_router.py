"""HTTP contract tests for the lung router.

The router must never let one model take the other down: a V2S failure returns
200 with ``v2s: null`` and the error string, and only a double failure is a 503.
Both pipelines are replaced by fakes, so no weight file is opened and no model
is built — the contract is exercised, not the weights.
"""

from __future__ import annotations

import io
from typing import Any, Optional

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.organs.lungs.classification import router as router_module
from app.organs.lungs.classification.pipeline import CLASS_NAMES, XRV_CLASS_NAMES


def _jpeg_bytes(seed: int = 0, size: int = 64) -> bytes:
    array = np.random.default_rng(seed).integers(0, 255, size=(size, size), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).convert("L").convert("RGB").save(buffer, "JPEG")
    return buffer.getvalue()


def _v2s_payload() -> dict[str, Any]:
    return {
        "top_class": "COVID",
        "confidence": 0.91,
        "probabilities": [0.91, 0.04, 0.03, 0.02],
        "class_names": list(CLASS_NAMES),
        "gradcam_base64": "iVBORw0KGgo=",
        "gradcam_grid": [300, 300],
        "bbox": [10, 20, 200, 240],
        "inference_ms": 12.5,
        "model_tag": "tf_efficientnetv2_s.in21k_ft_in1k",
        "infer_size": 300,
    }


def _xrv_payload(detail: bool = False) -> dict[str, Any]:
    return {
        "findings": [{"finding": "Effusion", "probability": 0.78, "rank": 1}],
        "threshold": 0.5,
        "count_above_threshold": 1,
        "scores": (
            [{"finding": name, "probability": 0.1} for name in XRV_CLASS_NAMES]
            if detail
            else None
        ),
        "gradcam_base64": "iVBORw0KGgo=" if detail else None,
        "inference_ms": 8.0,
        "model_tag": "xrv-densenet121-all",
    }


class _FakeModel:
    """Stands in for V2SPipeline / XRVPipeline: state and device only."""

    def __init__(self, loaded: bool = True, error: Optional[str] = None) -> None:
        self.is_loaded = loaded
        self.last_error = error
        self.device = "cpu"


class _FakeFacade:
    """Stands in for DualLungPipeline. ``fail`` names the models that raise."""

    def __init__(self, fail: tuple[str, ...] = ()) -> None:
        self.fail = set(fail)
        self.v2s = _FakeModel(loaded="v2s" not in self.fail)
        self.xrv = _FakeModel(loaded="xrv" not in self.fail)
        self.calls: list[tuple[str, Any]] = []

    def load(self, which: str = "all") -> dict[str, Optional[str]]:
        self.calls.append(("load", which))
        selected = ("v2s", "xrv") if which == "all" else (which,)
        errors: dict[str, Optional[str]] = {}
        for name in selected:
            if name in self.fail:
                errors[name] = f"RuntimeError: {name} weights exploded"
                setattr(self, name, _FakeModel(loaded=False, error=errors[name]))
            else:
                errors[name] = None
        return errors

    def unload(self, which: str = "all") -> dict[str, bool]:
        self.calls.append(("unload", which))
        selected = ("v2s", "xrv") if which == "all" else (which,)
        return dict.fromkeys(selected, True)

    def classify(self, image: Image.Image, include_xrv_detail: bool = False) -> dict[str, Any]:
        self.calls.append(("classify", include_xrv_detail))
        errors: dict[str, Optional[str]] = {"v2s": None, "xrv": None}
        v2s: Optional[dict[str, Any]] = None
        xrv: Optional[dict[str, Any]] = None
        if "v2s" in self.fail:
            errors["v2s"] = "RuntimeError: v2s forward failed"
        else:
            v2s = _v2s_payload()
        if "xrv" in self.fail:
            errors["xrv"] = "RuntimeError: xrv forward failed"
        else:
            xrv = _xrv_payload(include_xrv_detail)
        return {"v2s": v2s, "xrv": xrv, "errors": errors, "total_ms": 21.0}

    def health(self) -> dict[str, Any]:
        present = {"v2s": "v2s" not in self.fail, "xrv": "xrv" not in self.fail}
        status = "ok" if all(present.values()) else "degraded" if any(present.values()) else "unavailable"
        return {
            "status": status,
            "v2s": {"loaded": self.v2s.is_loaded, "checkpoint_present": present["v2s"], "detail": None},
            "xrv": {"loaded": self.xrv.is_loaded, "checkpoint_present": present["xrv"], "detail": None},
        }

    def model_info(self) -> dict[str, Any]:
        return {
            "v2s": {
                "model_tag": "tf_efficientnetv2_s.in21k_ft_in1k",
                "classes": list(CLASS_NAMES),
                "input_size": 300,
                "checkpoint": {"present": True, "loaded": self.v2s.is_loaded},
            },
            "xrv": {
                "model_tag": "xrv-densenet121-all",
                "classes": list(XRV_CLASS_NAMES),
                "input_size": 224,
                "checkpoint": {"present": True, "loaded": self.xrv.is_loaded},
            },
        }


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    from app.main import app

    monkeypatch.setattr(router_module, "_facade", _FakeFacade(), raising=False)
    return TestClient(app)


def _install(monkeypatch, facade: _FakeFacade) -> None:
    monkeypatch.setattr(router_module, "_facade", facade, raising=False)


def _post_scan(client: TestClient, **params):
    return client.post(
        "/api/lungs/classify",
        files={"file": ("scan.jpg", _jpeg_bytes(1), "image/jpeg")},
        params=params,
    )


# ─────────────────────────────────────────────────────────────────────────────
# /classify
# ─────────────────────────────────────────────────────────────────────────────


def test_both_models_ok_returns_200_with_both_payloads(client: TestClient) -> None:
    response = _post_scan(client)

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["v2s"]["top_class"] in CLASS_NAMES
    assert len(body["v2s"]["probabilities"]) == len(CLASS_NAMES)
    assert body["v2s"]["infer_size"] == 300
    assert body["v2s"]["gradcam_grid"] == [300, 300]

    assert body["xrv"]["threshold"] == 0.5
    assert body["xrv"]["count_above_threshold"] == 1
    assert body["xrv"]["findings"][0]["finding"] == "Effusion"
    # The default call must not ship all 18 scores or the XRV heatmap.
    assert body["xrv"]["scores"] is None
    assert body["xrv"]["gradcam_base64"] is None

    assert body["errors"] == {"v2s": None, "xrv": None}
    assert body["total_ms"] >= 0.0


@pytest.mark.parametrize("failing", ["v2s", "xrv"])
def test_one_model_failing_returns_200_with_the_other(
    client: TestClient, monkeypatch, failing: str
) -> None:
    _install(monkeypatch, _FakeFacade(fail=(failing,)))

    response = _post_scan(client)

    assert response.status_code == 200, response.text
    body = response.json()

    assert body[failing] is None
    assert failing in body["errors"][failing]
    assert "RuntimeError" in body["errors"][failing]

    healthy = "xrv" if failing == "v2s" else "v2s"
    assert body[healthy] is not None
    assert body["errors"][healthy] is None


def test_both_models_failing_returns_503(client: TestClient, monkeypatch) -> None:
    _install(monkeypatch, _FakeFacade(fail=("v2s", "xrv")))

    response = _post_scan(client)

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["message"] == "Both lung models failed."
    assert "v2s" in detail and "xrv" in detail
    assert "RuntimeError" in detail["v2s"]
    assert "RuntimeError" in detail["xrv"]


def test_include_xrv_detail_is_forwarded_to_the_pipeline(
    client: TestClient, monkeypatch
) -> None:
    facade = _FakeFacade()
    _install(monkeypatch, facade)

    response = _post_scan(client, include_xrv_detail="true")

    assert response.status_code == 200, response.text
    body = response.json()
    assert ("classify", True) in facade.calls
    assert len(body["xrv"]["scores"]) == len(XRV_CLASS_NAMES)
    assert body["xrv"]["gradcam_base64"] is not None


def test_default_call_does_not_request_xrv_detail(client: TestClient, monkeypatch) -> None:
    facade = _FakeFacade()
    _install(monkeypatch, facade)

    _post_scan(client)

    assert ("classify", False) in facade.calls


def test_classify_rejects_a_non_image_content_type(client: TestClient) -> None:
    response = client.post(
        "/api/lungs/classify",
        files={"file": ("scan.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_classify_rejects_a_bad_signature(client: TestClient) -> None:
    response = client.post(
        "/api/lungs/classify",
        files={"file": ("scan.jpg", b"%PDF-1.4 not an image", "image/jpeg")},
    )
    assert response.status_code == 415


# ─────────────────────────────────────────────────────────────────────────────
# /health, /model-info, /warmup, /unload
# ─────────────────────────────────────────────────────────────────────────────


def test_health_reports_both_models(client: TestClient) -> None:
    response = client.get("/api/lungs/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["v2s"]["loaded"] is True
    assert body["xrv"]["checkpoint_present"] is True


def test_health_reports_degraded_when_one_model_is_unavailable(
    client: TestClient, monkeypatch
) -> None:
    _install(monkeypatch, _FakeFacade(fail=("xrv",)))

    body = client.get("/api/lungs/health").json()

    assert body["status"] == "degraded"
    assert body["xrv"]["checkpoint_present"] is False


def test_model_info_names_both_label_spaces(client: TestClient) -> None:
    body = client.get("/api/lungs/model-info").json()

    assert body["v2s"]["classes"] == list(CLASS_NAMES)
    assert body["v2s"]["input_size"] == 300
    assert body["xrv"]["classes"] == list(XRV_CLASS_NAMES)
    assert body["xrv"]["input_size"] == 224
    assert body["xrv"]["model_tag"] == "xrv-densenet121-all"


def test_warmup_returns_partial_when_one_model_fails(
    client: TestClient, monkeypatch
) -> None:
    _install(monkeypatch, _FakeFacade(fail=("xrv",)))

    response = client.post("/api/lungs/warmup", json={"model": "all"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "partial"
    assert body["loaded"] == {"v2s": True, "xrv": False}
    assert body["errors"]["xrv"]


def test_warmup_returns_503_when_nothing_loads(client: TestClient, monkeypatch) -> None:
    _install(monkeypatch, _FakeFacade(fail=("v2s", "xrv")))

    response = client.post("/api/lungs/warmup", json={"model": "all"})

    assert response.status_code == 503
    assert response.json()["detail"]["message"] == "No lung model could be loaded."


def test_warmup_defaults_to_all(client: TestClient, monkeypatch) -> None:
    facade = _FakeFacade()
    _install(monkeypatch, facade)

    body = client.post("/api/lungs/warmup").json()

    assert body["requested"] == "all"
    assert ("load", "all") in facade.calls


def test_unload_forwards_the_selector(client: TestClient, monkeypatch) -> None:
    facade = _FakeFacade()
    _install(monkeypatch, facade)

    body = client.post("/api/lungs/unload", json={"model": "xrv"}).json()

    assert body["status"] == "ok"
    assert body["unloaded"] == {"xrv": True}
    assert ("unload", "xrv") in facade.calls
