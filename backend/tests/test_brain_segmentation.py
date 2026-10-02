"""Regression tests for the brain segmentation module (LiteMedSAM).

The critical regression here is the ``preprocess_image``/``postprocess_mask``
path: it used ``torch.nn.functional.interpolate`` without importing ``F``, so
every segmentation call NameError'd.  These tests exercise that exact path.

``verify_checkpoint`` / ``prepare_model`` are tested against a synthetic
random-weights checkpoint so the weight-preparation flow is proven without a
network download.  The minimum-size gate is patched away for the synthetic file.
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
    heatmap_to_point_label,
    heatmap_to_point_prompt,
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


def test_heatmap_to_box_prompt_none_for_saturated_cam() -> None:
    """An all-one map is flat too — a box around the whole frame means nothing."""
    saturated = np.ones((224, 224), dtype=np.float32)
    assert heatmap_to_box_prompt(saturated, resized_hw=(256, 224), adaptive=True) is None


def test_heatmap_box_default_is_the_fixed_threshold_path() -> None:
    """The adaptive path is opt-in; the default must stay the validated one.

    The adaptive box measured ~3× looser and made the decoder segment well
    beyond the lesion, so a default that quietly drifts to it is a regression.
    """
    cam = np.zeros((224, 224), dtype=np.float32)
    cam[80:150, 60:130] = 0.9

    default_box = heatmap_to_box_prompt(cam, resized_hw=(256, 256))
    explicit_fixed = heatmap_to_box_prompt(cam, resized_hw=(256, 256), threshold=0.5, margin_px=5)

    assert default_box is not None and explicit_fixed is not None
    assert torch.equal(default_box, explicit_fixed), (
        "default path is no longer the fixed-threshold box"
    )


def test_pipeline_default_box_uses_the_production_threshold_and_margin(
    synthetic_checkpoint: str,
    sample_image: Image.Image,
    hot_cam,
) -> None:
    """Pin the served box parameters, not just the helper's defaults.

    The pipeline passes its own threshold and margin, so the helper's defaults
    agreeing is not enough — this is the combination the workspace runs, and it
    is the one that regressed when the box builder changed underneath it.
    """
    pipe = BrainSegmentationPipeline(
        model_path=synthetic_checkpoint, device="cpu"
    )

    res = pipe.segment_with_heatmap(sample_image, hot_cam)

    expected = heatmap_to_box_prompt(
        hot_cam, resized_hw=(256, 256), threshold=0.5, margin_px=10
    )
    assert expected is not None
    assert res["box_prompt_used"] is True
    assert res["box_coords"] == expected.squeeze().tolist()

    pipe.unload()


def test_adaptive_box_is_looser_than_the_fixed_box() -> None:
    """Documents why adaptive is not the default: it is materially looser."""
    cam = np.zeros((224, 224), dtype=np.float32)
    cam[80:150, 60:130] = 0.9

    fixed = heatmap_to_box_prompt(cam, resized_hw=(256, 256)).squeeze().tolist()
    adaptive = heatmap_to_box_prompt(cam, resized_hw=(256, 256), adaptive=True).squeeze().tolist()

    fixed_area = (fixed[2] - fixed[0]) * (fixed[3] - fixed[1])
    adaptive_area = (adaptive[2] - adaptive[0]) * (adaptive[3] - adaptive[1])
    assert adaptive_area > fixed_area


def test_adaptive_box_finds_a_region_a_fixed_threshold_would_miss() -> None:
    """The reason the adaptive path exists at all: amplitude varies by scan.

    A fixed 0.5 sees nothing here; the adaptive cut still finds the lesion.
    """
    cam = np.zeros((224, 224), dtype=np.float32)
    cam[60:160, 60:160] = 0.30  # peak well under any fixed 0.5 cut

    assert heatmap_to_box_prompt(cam, resized_hw=(256, 256), threshold=0.5) is None

    adaptive = heatmap_to_box_prompt(cam, resized_hw=(256, 256), adaptive=True)
    assert adaptive is not None
    x1, y1, x2, y2 = adaptive.squeeze().tolist()
    assert x2 - x1 > 0 and y2 - y1 > 0


def test_adaptive_box_keeps_the_largest_component_and_drops_satellites() -> None:
    """A small high-amplitude blob must not win over the lesion."""
    cam = np.zeros((224, 224), dtype=np.float32)
    cam[40:120, 40:120] = 0.90   # the lesion
    cam[200:210, 200:210] = 1.00  # brighter, but tiny — skull/eye-orbit style

    box = heatmap_to_box_prompt(cam, resized_hw=(256, 256), adaptive=True)
    assert box is not None
    x1, y1, x2, y2 = box.squeeze().tolist()

    # The satellite sits near 228+ in 256-space; the box must stay well clear.
    assert x2 < 200 and y2 < 200, f"satellite activation pulled the box to {box}"
    assert x1 < 100 and y1 < 100


def test_adaptive_box_margin_grows_the_box() -> None:
    cam = np.zeros((224, 224), dtype=np.float32)
    cam[80:140, 90:150] = 0.9

    tight = heatmap_to_box_prompt(cam, resized_hw=(256, 256), adaptive=True, margin_frac=0.0)
    padded = heatmap_to_box_prompt(cam, resized_hw=(256, 256), adaptive=True, margin_frac=0.10)

    assert tight is not None and padded is not None
    tx1, ty1, tx2, ty2 = tight.squeeze().tolist()
    px1, py1, px2, py2 = padded.squeeze().tolist()

    assert (px2 - px1) > (tx2 - tx1)
    assert (py2 - py1) > (ty2 - ty1)
    assert px1 <= tx1 and py1 <= ty1
    assert px2 >= tx2 and py2 >= ty2


def test_adaptive_box_never_collapses_to_a_line() -> None:
    """SAM needs a usable box; a degenerate one must be expanded, not returned."""
    cam = np.zeros((224, 224), dtype=np.float32)
    cam[100:114, 100:114] = 0.9  # survives morphological opening, but is small

    box = heatmap_to_box_prompt(cam, resized_hw=(256, 256), adaptive=True)
    assert box is not None
    x1, y1, x2, y2 = box.squeeze().tolist()
    assert x2 > x1 and y2 > y1
    assert min(x2 - x1, y2 - y1) >= 8


def test_heatmap_point_prompt_sits_on_the_peak() -> None:
    cam = np.zeros((224, 224), dtype=np.float32)
    cam[120, 150] = 1.0  # row 120, col 150

    point = heatmap_to_point_prompt(cam)
    assert point is not None
    assert point.shape == (1, 1, 2)
    x, y = point.squeeze().tolist()
    # 224 -> 256 rescales by 256/224; the peak must land near that, not at origin.
    assert x == pytest.approx(150 * 256 / 224, abs=3)
    assert y == pytest.approx(120 * 256 / 224, abs=3)


def test_heatmap_point_prompt_none_for_flat_cam() -> None:
    assert heatmap_to_point_prompt(np.zeros((224, 224), dtype=np.float32)) is None


def test_heatmap_point_label_is_positive() -> None:
    assert heatmap_to_point_label().tolist() == [[1]]


def test_click_box_is_always_full_size_and_contains_the_click(
    synthetic_checkpoint: str,
    sample_image: Image.Image,
) -> None:
    """Edge clicks must shift the box, not shrink it.

    Clamping both edges inward produced a half-size box for clicks near the
    image border, and smaller boxes cost Dice on the box-quality curve. The box
    must stay the requested size and keep the click inside it.
    """
    pipe = BrainSegmentationPipeline(
        model_path=synthetic_checkpoint, device="cpu"
    )

    side = 48
    for cx, cy in ((5.0, 5.0), (250.0, 250.0), (128.0, 128.0), (0.0, 255.0)):
        res = pipe.segment_with_click(sample_image, cx, cy, box_size=side)
        x1, y1, x2, y2 = res["box_coords"]

        assert (x2 - x1) == pytest.approx(side), f"box shrank for click ({cx},{cy})"
        assert (y2 - y1) == pytest.approx(side)
        assert 0 <= x1 <= 255 and 0 <= x2 <= 255
        assert 0 <= y1 <= 255 and 0 <= y2 <= 255
        # The click must sit inside the box that was built from it.
        assert x1 <= cx <= x2 and y1 <= cy <= y2

    # A click away from any edge is centred, as expected.
    centred = pipe.segment_with_click(sample_image, 128.0, 128.0, box_size=side)
    assert centred["box_coords"] == [104.0, 104.0, 152.0, 152.0]

    pipe.unload()


def test_click_box_rejects_an_unusable_size(
    synthetic_checkpoint: str,
    sample_image: Image.Image,
) -> None:
    pipe = BrainSegmentationPipeline(
        model_path=synthetic_checkpoint, device="cpu"
    )

    for bad in (8, 500):
        with pytest.raises(ValueError, match="box_size must be between"):
            pipe.segment_with_click(sample_image, 128.0, 128.0, box_size=bad)

    pipe.unload()


def test_click_segment_reports_click_box_mode(
    synthetic_checkpoint: str,
    sample_image: Image.Image,
) -> None:
    """The click path tags its output so the UI can label it distinctly."""
    pipe = BrainSegmentationPipeline(
        model_path=synthetic_checkpoint, device="cpu"
    )

    res = pipe.segment_with_click(sample_image, 100.0, 100.0, box_size=48)

    assert res["prompt_mode"] == "click_box"
    assert res["click"] == [100.0, 100.0]
    assert res["segmentation_performed"] is True
    assert res["seg_mask_base64"].startswith("data:image/png;base64,")
    assert res["seg_overlay_base64"].startswith("data:image/jpeg;base64,")
    assert isinstance(res["mask_foreground_px"], int)
    assert res["mask_foreground_px"] >= 0

    pipe.unload()


# ── prepare_weights ───────────────────────────────────────────────────────


def test_verify_checkpoint_reports_ready_strict(synthetic_checkpoint: str) -> None:
    from app.organs.brain.segmentation.model import build_medsam_lite

    # Patch the min-size gate so the small synthetic file passes shape checks.
    prepare_weights.EXPECTED_MIN_BYTES = 0
    try:
        report = prepare_weights.verify_checkpoint(Path(synthetic_checkpoint), device="cpu")
    finally:
        prepare_weights.EXPECTED_MIN_BYTES = 30_000_000

    assert report["parameters"] == sum(
        p.numel() for p in build_medsam_lite().parameters()
    )
    # Box prompt only — the released LiteMedSAM weights are box-only, so the
    # readiness forward pass no longer exercises a dense mask prompt.
    assert report["forward_shapes"]["box"] == (1, 1, 256, 256)
    assert "mask" not in report["forward_shapes"]


def test_verify_checkpoint_rejects_garbage(tmp_path) -> None:
    garbage = tmp_path / "garbage.pth"
    garbage.write_bytes(b"this is not a torch checkpoint" * 100)

    prepare_weights.EXPECTED_MIN_BYTES = 0
    try:
        with pytest.raises(RuntimeError):
            prepare_weights.verify_checkpoint(garbage, device="cpu")
    finally:
        prepare_weights.EXPECTED_MIN_BYTES = 30_000_000


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
    # The fixture (352px longest side) sits under SEG_MAX_SIDE, so the mask
    # comes back at the original size. The downscale cap is covered separately.
    assert mask.shape == (h, w)

    pipe.unload()
    assert pipe.is_loaded is False


def test_segmentation_output_is_capped_for_the_browser(
    synthetic_checkpoint: str,
) -> None:
    """Regression: mask/overlay data URLs were encoded at full input
    resolution, so a large upload produced megabyte-sized base64 payloads that
    made the response heavy and the page scroll janky. Rendered output is now
    downscaled to ``SEG_MAX_SIDE`` on its longest side before encoding."""
    from app.organs.brain.segmentation.pipeline import SEG_MAX_SIDE

    big = Image.fromarray(np.full((900, 700, 3), 30, dtype=np.uint8), "RGB")
    pipe = BrainSegmentationPipeline(model_path=synthetic_checkpoint, device="cpu")

    res = pipe.segment_with_box(big, (100.0, 100.0, 600.0, 700.0))

    assert res["original_size"] == "700x900"

    mask = np.asarray(
        Image.open(
            _io.BytesIO(_b64.b64decode(res["box_mask_base64"].split(",", 1)[1]))
        ).convert("L")
    )
    assert max(mask.shape) == SEG_MAX_SIDE

    overlay = np.asarray(
        Image.open(
            _io.BytesIO(_b64.b64decode(res["box_overlay_base64"].split(",", 1)[1]))
        )
    )
    assert max(overlay.shape[:2]) == SEG_MAX_SIDE

    pipe.unload()


def test_pipeline_segment_with_heatmap_end_to_end(
    synthetic_checkpoint: str,
    sample_image: Image.Image,
    hot_cam,
) -> None:
    """The exact router path: heatmap → box prompt → box mask."""
    pipe = BrainSegmentationPipeline(
        model_path=synthetic_checkpoint, device="cpu"
    )

    res = pipe.segment_with_heatmap(sample_image, hot_cam)

    assert res["segmentation_performed"] is True
    assert res["box_prompt_used"] is True
    assert res["box_mask_base64"].startswith("data:image/png;base64,")
    assert res["box_overlay_base64"].startswith("data:image/jpeg;base64,")
    assert isinstance(res["iou_box"], float)
    assert "mask_prompt" not in res  # box-only pipeline

    pipe.unload()


def test_pipeline_segment_with_heatmap_skips_when_flat_cam(
    synthetic_checkpoint: str,
    sample_image: Image.Image,
) -> None:
    """No heatmap activation → no box → segmentation reports itself skipped."""
    pipe = BrainSegmentationPipeline(
        model_path=synthetic_checkpoint, device="cpu"
    )
    flat = np.zeros((224, 224), dtype=np.float32)

    res = pipe.segment_with_heatmap(sample_image, flat)

    assert res["segmentation_performed"] is True
    assert res["box_prompt_used"] is False
    assert res.get("box_mask_base64") is None
    assert "no region" in res["box_prompt_note"].lower()

    pipe.unload()


def test_pipeline_segment_with_heatmap_reports_its_prompt_mode(
    synthetic_checkpoint: str,
    sample_image: Image.Image,
    hot_cam,
) -> None:
    """The heatmap path must tag its output so the API can label the mask."""
    pipe = BrainSegmentationPipeline(
        model_path=synthetic_checkpoint, device="cpu"
    )

    res = pipe.segment_with_heatmap(sample_image, hot_cam)

    assert res["prompt_mode"] == "heatmap_box"

    pipe.unload()


def test_pipeline_segment_raw_end_to_end(
    synthetic_checkpoint: str,
    sample_image: Image.Image,
) -> None:
    """The unprompted path produces a full mask/overlay payload like any other."""
    pipe = BrainSegmentationPipeline(
        model_path=synthetic_checkpoint, device="cpu"
    )

    res = pipe.segment_raw(sample_image)

    assert res["segmentation_performed"] is True
    assert res["prompt_mode"] == "raw"
    assert res["seg_mask_base64"].startswith("data:image/png;base64,")
    assert res["seg_overlay_base64"].startswith("data:image/jpeg;base64,")
    assert isinstance(res["iou_pred"], float)
    # A count is always reported, so a zero-pixel mask is distinguishable from a
    # missing one. Synthetic weights may legitimately yield either.
    assert isinstance(res["mask_foreground_px"], int)
    assert res["mask_foreground_px"] >= 0
    # Raw mode is not a box prompt — none of the box-specific keys may appear.
    assert "box_prompt_used" not in res
    assert "box_coords" not in res

    mask = np.asarray(
        Image.open(
            _io.BytesIO(_b64.b64decode(res["seg_mask_base64"].split(",", 1)[1]))
        )
    )
    # postprocess_mask crops back to the original image geometry.
    assert mask.shape[:2] == (288, 352)

    pipe.unload()


def test_pipeline_segment_raw_sends_the_model_no_prompt_at_all(
    synthetic_checkpoint: str,
    sample_image: Image.Image,
) -> None:
    """The whole point of raw mode: nothing reaches the prompt encoder.

    Asserted on the prompt encoder rather than on the output, because a
    *silently* box-prompted forward would still return a plausible-looking mask
    and the ablation would be meaningless.
    """
    pipe = BrainSegmentationPipeline(
        model_path=synthetic_checkpoint, device="cpu"
    )

    seen: dict[str, object] = {}
    original_forward = pipe.model.prompt_encoder.forward

    def spy(points=None, boxes=None, masks=None):
        seen["points"] = points
        seen["boxes"] = boxes
        seen["masks"] = masks
        return original_forward(points=points, boxes=boxes, masks=masks)

    pipe.model.prompt_encoder.forward = spy  # type: ignore[method-assign]

    pipe.segment_raw(sample_image)

    assert seen == {"points": None, "boxes": None, "masks": None}

    pipe.unload()


def test_heatmap_prompt_sends_no_point_by_default(
    synthetic_checkpoint: str,
    sample_image: Image.Image,
    hot_cam,
) -> None:
    """The peak point is opt-in.

    Sending it unconditionally measured 0.051 worse mean IoU across the eight
    specimen samples, so the default must stay off. Pinned here because a
    default that quietly drifts back on is invisible in a result.
    """
    pipe = BrainSegmentationPipeline(
        model_path=synthetic_checkpoint, device="cpu"
    )

    seen: dict[str, object] = {}
    original_forward = pipe.model.prompt_encoder.forward

    def spy(points=None, boxes=None, masks=None):
        seen["points"] = points
        seen["boxes"] = boxes
        return original_forward(points=points, boxes=boxes, masks=masks)

    pipe.model.prompt_encoder.forward = spy  # type: ignore[method-assign]

    res = pipe.segment_with_heatmap(sample_image, hot_cam)

    assert seen["points"] is None, "default run sent a point prompt"
    assert seen["boxes"] is not None, "default run should still send the box"
    assert res["point_prompt_used"] is False
    assert "point_coords" not in res

    pipe.unload()


def test_heatmap_prompt_can_opt_into_the_point_prompt(
    synthetic_checkpoint: str,
    sample_image: Image.Image,
    hot_cam,
) -> None:
    """The point prompt still works when explicitly requested."""
    pipe = BrainSegmentationPipeline(
        model_path=synthetic_checkpoint, device="cpu"
    )

    seen: dict[str, object] = {}
    original_forward = pipe.model.prompt_encoder.forward

    def spy(points=None, boxes=None, masks=None):
        seen["points"] = points
        return original_forward(points=points, boxes=boxes, masks=masks)

    pipe.model.prompt_encoder.forward = spy  # type: ignore[method-assign]

    res = pipe.segment_with_heatmap(sample_image, hot_cam, use_peak_point=True)

    points = seen["points"]
    assert points is not None, "opt-in run did not send a point prompt"
    coords, labels = points
    assert coords.shape == (1, 1, 2)
    assert labels.tolist() == [[1]]  # 1 = positive point
    assert res["point_prompt_used"] is True
    assert len(res["point_coords"]) == 2

    pipe.unload()