"""Serving-contract tests: the two preprocessing paths and the sidecar guard.

LungV2S and XRV are fed by two different input distributions. Crossing them
raises nothing — both networks happily return a confident, meaningless answer —
so the only defence is that the two paths are tested to produce *different*
tensors with *different* ranges, and that neither can be reached through the
other's function.

No checkpoint is opened here. The pipeline shells are built and their
preprocessors driven directly, which is exactly the code that runs per request.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from app.organs.lungs.classification import model_v2s
from app.organs.lungs.classification.model_v2s import (
    CLASS_NAMES,
    SUPPORTED_INFER_SIZES,
    load_sidecar,
    preprocess_v2s,
    resolve_sidecar,
)
from app.organs.lungs.classification.pipeline import (
    XRV_CLASS_NAMES,
    XRV_INPUT_SIZE,
    V2SPipeline,
    XRVPipeline,
)


def _scan(size: int = 512, seed: int = 0) -> Image.Image:
    """An image that spans the full 0..255 range, so both ends are exercised."""
    ramp = np.linspace(0, 255, size, dtype=np.float32)
    array = np.tile(ramp, (size, 1)).astype(np.uint8)
    array[: size // 8, : size // 8] = 0
    array[-size // 8 :, -size // 8 :] = 255
    return Image.fromarray(array).convert("L")


# ─────────────────────────────────────────────────────────────────────────────
# LungV2S preprocessing
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("infer_size", SUPPORTED_INFER_SIZES)
def test_v2s_preprocess_shape_channels_and_range(infer_size: int) -> None:
    tensor = preprocess_v2s(_scan(), infer_size)

    assert tuple(tensor.shape) == (1, 3, infer_size, infer_size)
    assert tensor.dtype == torch.float32

    values = tensor.numpy()
    assert values.min() >= -2.2
    assert values.max() <= 2.7
    # Both ends of the ImageNet normalisation are actually reached.
    assert values.min() < -2.0
    assert values.max() > 2.5


def test_v2s_preprocess_replicates_one_luminance_plane() -> None:
    tensor = preprocess_v2s(_scan(), 300)
    mean = torch.tensor(model_v2s.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(model_v2s.IMAGENET_STD).view(1, 3, 1, 1)

    pixels = (tensor * std + mean).numpy()  # back to 0..1 RGB

    assert np.allclose(pixels[0, 0], pixels[0, 1], atol=1e-6)
    assert np.allclose(pixels[0, 0], pixels[0, 2], atol=1e-6)
    assert pixels.min() >= -1e-6
    assert pixels.max() <= 1.0 + 1e-6


def test_v2s_preprocess_rejects_an_unsupported_size() -> None:
    with pytest.raises(ValueError, match="infer_size must be one of"):
        preprocess_v2s(_scan(), 256)


def test_v2s_preprocess_rejects_non_pil_input() -> None:
    with pytest.raises(TypeError, match="expects a PIL image"):
        preprocess_v2s(np.zeros((8, 8), dtype=np.uint8), 300)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# XRV preprocessing
# ─────────────────────────────────────────────────────────────────────────────


def test_xrv_preprocess_shape_channels_and_range(tmp_path: Path) -> None:
    pipeline = XRVPipeline(cache_dir=str(tmp_path / "xrv"))

    tensor = pipeline._preprocess(_scan())

    assert tuple(tensor.shape) == (1, 1, XRV_INPUT_SIZE, XRV_INPUT_SIZE)
    assert tensor.dtype == torch.float32
    values = tensor.numpy()
    assert values.min() == pytest.approx(-1024.0)
    assert values.max() == pytest.approx(1024.0)
    assert values.min() >= -1024.0
    assert values.max() <= 1024.0


def test_xrv_preprocess_is_single_channel_grayscale(tmp_path: Path) -> None:
    pipeline = XRVPipeline(cache_dir=str(tmp_path / "xrv"))
    tensor = pipeline._preprocess(_scan())
    assert tensor.shape[1] == 1
    # A 0..255 ramp must come back as a ramp, not as ImageNet-normalised values.
    assert float(tensor.abs().max()) > 1000.0


def test_xrv_class_order_is_the_18_pathologies() -> None:
    assert len(XRV_CLASS_NAMES) == 18
    assert XRV_CLASS_NAMES[0] == "Atelectasis"
    assert XRV_CLASS_NAMES[-1] == "Enlarged Cardiomediastinum"
    assert len(set(XRV_CLASS_NAMES)) == 18


# ─────────────────────────────────────────────────────────────────────────────
# The two paths are not interchangeable
# ─────────────────────────────────────────────────────────────────────────────


def test_preprocessings_are_not_interchangeable(tmp_path: Path) -> None:
    image = _scan()
    v2s = preprocess_v2s(image, 300)
    xrv = XRVPipeline(cache_dir=str(tmp_path / "xrv"))._preprocess(image)

    # Different rank, different resolution, different channel count.
    assert tuple(v2s.shape) == (1, 3, 300, 300)
    assert tuple(xrv.shape) == (1, 1, 224, 224)
    assert v2s.shape[1] != xrv.shape[1]

    # Different numeric range, by three orders of magnitude.
    v2s_min, v2s_max = float(v2s.min()), float(v2s.max())
    xrv_min, xrv_max = float(xrv.min()), float(xrv.max())
    assert (v2s_min, v2s_max) != pytest.approx((xrv_min, xrv_max))
    assert abs(v2s_min) < 3.0 and abs(v2s_max) < 3.0
    assert abs(xrv_min) > 1000.0 and abs(xrv_max) > 1000.0

    # A tensor from one path fails the other's range contract outright.
    assert not (xrv_min >= -2.2 and xrv_max <= 2.7)
    assert not (v2s_min <= -1000.0 and v2s_max >= 1000.0)


def test_pipeline_exposes_the_two_paths_on_different_objects(tmp_path: Path) -> None:
    """Each pipeline owns its own preprocessing; neither calls the other's."""
    v2s_pipeline = V2SPipeline(checkpoint=str(tmp_path / "absent.pth"))
    xrv_pipeline = XRVPipeline(cache_dir=str(tmp_path / "xrv"))

    assert not hasattr(v2s_pipeline, "_preprocess")
    assert hasattr(xrv_pipeline, "_preprocess")
    assert xrv_pipeline._preprocess(_scan()).shape[1] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Sidecar
# ─────────────────────────────────────────────────────────────────────────────


def _sidecar_payload(**overrides) -> dict:
    payload = {
        "class_names": list(CLASS_NAMES),
        "num_classes": len(CLASS_NAMES),
        "model_tag": model_v2s.MODEL_TAG,
        "input_size": 300,
        "infer_size": 300,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "metrics": {},
        "thresholds": [0.5] * len(CLASS_NAMES),
        "split": {"train": 0, "val": 0, "test": 0, "strategy": "phash-cluster-grouped"},
    }
    payload.update(overrides)
    return payload


def _write_sidecar(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_sidecar_round_trips_a_supported_infer_size(tmp_path: Path) -> None:
    path = _write_sidecar(tmp_path / "lung_v2s.label_space.json", _sidecar_payload(infer_size=384))

    sidecar = load_sidecar(str(path))

    assert sidecar["infer_size"] == 384
    assert sidecar["input_size"] == 300
    assert sidecar["class_names"] == list(CLASS_NAMES)
    assert sidecar["num_classes"] == len(CLASS_NAMES)


def test_sidecar_refuses_a_different_class_order(tmp_path: Path) -> None:
    path = _write_sidecar(
        tmp_path / "lung_v2s.label_space.json",
        _sidecar_payload(class_names=["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]),
    )

    with pytest.raises(ValueError, match="does not match the serving order"):
        load_sidecar(str(path))


def test_sidecar_refuses_an_unsupported_infer_size(tmp_path: Path) -> None:
    path = _write_sidecar(tmp_path / "lung_v2s.label_space.json", _sidecar_payload(infer_size=256))

    with pytest.raises(ValueError, match="not one of"):
        load_sidecar(str(path))


def test_sidecar_refuses_a_different_head_width(tmp_path: Path) -> None:
    path = _write_sidecar(tmp_path / "lung_v2s.label_space.json", _sidecar_payload(num_classes=16))

    with pytest.raises(ValueError, match="this build serves"):
        load_sidecar(str(path))


def test_sidecar_requires_a_path() -> None:
    with pytest.raises(ValueError, match="requires a path"):
        load_sidecar(None)


def test_resolve_sidecar_finds_the_file_beside_or_above_the_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "lung_v2s_best.pth"
    checkpoint.parent.mkdir(parents=True)

    beside = _write_sidecar(tmp_path / "checkpoints" / "lung_v2s.label_space.json", _sidecar_payload())
    assert Path(resolve_sidecar(str(checkpoint))) == beside

    beside.unlink()
    above = _write_sidecar(tmp_path / "lung_v2s.label_space.json", _sidecar_payload())
    assert Path(resolve_sidecar(str(checkpoint))) == above


# ─────────────────────────────────────────────────────────────────────────────
# Fail-closed lifecycle (no weight files exist here)
# ─────────────────────────────────────────────────────────────────────────────


def test_v2s_pipeline_refuses_to_load_a_missing_checkpoint(tmp_path: Path) -> None:
    pipeline = V2SPipeline(checkpoint=str(tmp_path / "absent.pth"))

    assert pipeline.is_loaded is False
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        pipeline.load()
    assert pipeline.is_loaded is False


def test_xrv_pipeline_refuses_to_load_missing_weights(tmp_path: Path) -> None:
    pipeline = XRVPipeline(cache_dir=str(tmp_path / "empty"))

    assert pipeline.is_loaded is False
    with pytest.raises(FileNotFoundError, match="weights not found"):
        pipeline.load()
    assert pipeline.is_loaded is False


def test_checkpoint_info_is_honest_before_loading(tmp_path: Path) -> None:
    info = V2SPipeline(checkpoint=str(tmp_path / "absent.pth")).checkpoint_info()

    assert info["present"] is False
    assert info["loaded"] is False
    assert info["path"].endswith("absent.pth")
    assert "size_mb" not in info


def test_unload_on_an_unloaded_pipeline_is_a_no_op(tmp_path: Path) -> None:
    pipeline = V2SPipeline(checkpoint=str(tmp_path / "absent.pth"))
    pipeline.unload()
    assert pipeline.is_loaded is False
    assert os.path.basename(pipeline.checkpoint_path) == "absent.pth"
