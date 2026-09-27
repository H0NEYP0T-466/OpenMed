"""Regression tests for the brain segmentation module (LiteMedSAM).

The critical regression here is the ``preprocess_image``/``postprocess_mask``
path: it used ``torch.nn.functional.interpolate`` without importing ``F``, so
every segmentation call NameError'd.  These tests exercise that exact path.

``verify_checkpoint`` / ``prepare_model`` are tested against a synthetic
random-weights checkpoint so the weight-preparation flow is proven without a
network download.  The 80 MB size gate is patched away for the synthetic file.
"""

from __future__ import annotations

import base64 as _b64
import io as _io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from app.organs.brain.segmentation import prepare_weights
from app.organs.brain.segmentation.pipeline import (
    BrainSegmentationPipeline,
    SegmentationUnavailableError,
)
from app.organs.brain.segmentation.preprocessor import (
    MEDSAM_INPUT_SIZE,
    heatmap_to_box_prompt,
    heatmap_to_mask_prompt,
    postprocess_mask,
    preprocess_image,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def synthetic_checkpoint(tmp_path_factory) -> str:
    """A real LiteMedSAM state dict saved with random weights — no network.

    This is a *plausible* checkpoint: same architecture, same state-dict keys,
    so ``verify_checkpoint`` and ``BrainSegmentationPipeline._load`` treat it
    exactly like the real ``lite_medsam.pth``.
    """
    from app.organs.brain.segmentation.model import build_medsam_lite

    model = build_medsam_lite()
    path = tmp_path_factory.mktemp("checkpoints") / "lite_medsam.pth"
    torch.save(model.state_dict(), path)
    return str(path)


@pytest.fixture
def sample_image() -> Image.Image:
    """Small greyscale MRI-ish image: darker background, bright hotspot."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 60, size=(288, 352, 3), dtype=np.uint8)
    arr[90:170, 130:230, :] = np.clip(arr[90:170, 130:230, :].astype(int) + 160, 0, 255)
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def hot_cam() -> np.ndarray:
    """Grad-CAM-style heatmap with a single strong activation region."""
    cam = np.zeros((224, 224), dtype=np.float32)
    cam[80:150, 60:130] = 0.9
    return cam


# ── Preprocessor ──────────────────────────────────────────────────────────


def test_preprocess_image_geometry(sample_image: Image.Image) -> None:
    h, w = sample_image.size[1], sample_image.size[0]
    prep = preprocess_image(sample_image, MEDSAM_INPUT_SIZE)

    assert prep.tensor.shape == (1, 3, 256, 256)
    assert prep.original_hw == (h, w)
    # Longest side (width 352) scaled to 256 → height rounds to 209.
    # resized_hw is (H, W), i.e. (209, 256).
    assert prep.resized_hw == (209, 256)
    assert prep.scale_factor == pytest.approx(256 / 352)
    # Padded region stays zero, values in [0, 1]
    assert prep.tensor.min() >= 0.0 and prep.tensor.max() <= 1.0


def test_postprocess_mask_returns_original_size_binary(
    sample_image: Image.Image,
) -> None:
    """Regression: this path NameError'd on a missing ``F`` import."""
    prep = preprocess_image(sample_image, MEDSAM_INPUT_SIZE)

    low_res = torch.rand(1, 1, 64, 64)
    mask = postprocess_mask(low_res, prep.original_hw, prep.resized_hw)

    assert mask.shape == prep.original_hw
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 255}


def test_heatmap_to_box_prompt_shape(hot_cam: np.ndarray) -> None:
    box = heatmap_to_box_prompt(hot_cam, resized_hw=(256, 209), threshold=0.5)
    assert box is not None
    assert box.shape == (1, 1, 4)
    x1, y1, x2, y2 = box.squeeze().tolist()
    assert 0 <= x1 < x2 <= 255 and 0 <= y1 < y2 <= 255


def test_heatmap_to_box_prompt_none_for_flat_cam() -> None:
    flat = np.zeros((224, 224), dtype=np.float32)
    assert heatmap_to_box_prompt(flat, resized_hw=(256, 209)) is None


def test_heatmap_to_mask_prompt_shape(hot_cam: np.ndarray) -> None:
    mask_prompt = heatmap_to_mask_prompt(hot_cam, target_size=MEDSAM_INPUT_SIZE)
    assert mask_prompt.shape == (1, 1, 256, 256)
    assert mask_prompt.sum().item() > 0  # hotspot survived thresholding


# ── prepare_weights ───────────────────────────────────────────────────────


def test_verify_checkpoint_reports_ready_strict(synthetic_checkpoint: str) -> None:
    from app.organs.brain.segmentation.model import build_medsam_lite

    # Patch the 80 MB gate so the small synthetic file passes shape checks.
    prepare_weights.EXPECTED_MIN_BYTES = 0
    try:
        report = prepare_weights.verify_checkpoint(Path(synthetic_checkpoint), device="cpu")
    finally:
        prepare_weights.EXPECTED_MIN_BYTES = 80_000_000

    assert report["parameters"] == sum(
        p.numel() for p in build_medsam_lite().parameters()
    )
    assert report["forward_shapes"]["box"] == (1, 1, 256, 256)
    assert report["forward_shapes"]["mask"] == (1, 1, 256, 256)


def test_verify_checkpoint_rejects_garbage(tmp_path) -> None:
    garbage = tmp_path / "garbage.pth"
    garbage.write_bytes(b"this is not a torch checkpoint" * 100)

    prepare_weights.EXPECTED_MIN_BYTES = 0
    try:
        with pytest.raises(RuntimeError):
            prepare_weights.verify_checkpoint(garbage, device="cpu")
    finally:
        prepare_weights.EXPECTED_MIN_BYTES = 80_000_000


def test_prepare_model_no_download_when_present(
    synthetic_checkpoint: str, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(prepare_weights, "EXPECTED_MIN_BYTES", 0)

    dest = tmp_path / "installed" / "lite_medsam.pth"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(Path(synthetic_checkpoint).read_bytes())

    path, info = prepare_weights.prepare_model(
        model_path=dest, force=False, device="cpu"
    )
    assert path == dest
    assert info["forward_shapes"]["box"] == (1, 1, 256, 256)


# ── Pipeline end-to-end ───────────────────────────────────────────────────


def test_pipeline_not_loaded_raises(sample_image: Image.Image, hot_cam) -> None:
    pipe = BrainSegmentationPipeline()
    with pytest.raises(SegmentationUnavailableError):
        pipe.segment_with_heatmap(sample_image, hot_cam)
    with pytest.raises(SegmentationUnavailableError):
        pipe.segment_with_box(sample_image, (20.0, 20.0, 200.0, 200.0))


def test_pipeline_missing_checkpoint_raises(tmp_path) -> None:
    missing = str(tmp_path / "nope.pth")
    with pytest.raises(SegmentationUnavailableError, match="not found"):
        BrainSegmentationPipeline(model_path=missing)


def test_pipeline_segment_with_box_end_to_end(
    synthetic_checkpoint: str,
    sample_image: Image.Image,
) -> None:
    pipe = BrainSegmentationPipeline(model_path=synthetic_checkpoint, device="cpu")
    h, w = sample_image.size[1], sample_image.size[0]

    res = pipe.segment_with_box(sample_image, (20.0, 20.0, 300.0, 250.0))

    assert res["segmentation_performed"] is True
    assert res["box_mask_base64"].startswith("data:image/png;base64,")
    assert res["box_overlay_base64"].startswith("data:image/jpeg;base64,")
    assert res["input_size"] == "256x256"
    assert res["original_size"] == f"{w}x{h}"

    raw = _b64.b64decode(res["box_mask_base64"].split(",", 1)[1])
    mask = np.asarray(Image.open(_io.BytesIO(raw)).convert("L"))
    assert mask.shape == (h, w)

    pipe.unload()
    assert pipe.is_loaded is False


def test_pipeline_segment_with_heatmap_end_to_end(
    synthetic_checkpoint: str,
    sample_image: Image.Image,
    hot_cam,
) -> None:
    """The exact router path: heatmap → box prompt + mask prompt → both masks."""
    pipe = BrainSegmentationPipeline(
        model_path=synthetic_checkpoint, device="cpu"
    )

    res = pipe.segment_with_heatmap(sample_image, hot_cam)

    assert res["segmentation_performed"] is True
    assert res["box_prompt_used"] is True
    assert res["mask_prompt_used"] is True
    assert res["box_mask_base64"].startswith("data:image/png;base64,")
    assert res["mask_prompt_mask_base64"].startswith("data:image/png;base64,")
    assert isinstance(res["iou_box"], float) and isinstance(res["iou_mask"], float)

    pipe.unload()