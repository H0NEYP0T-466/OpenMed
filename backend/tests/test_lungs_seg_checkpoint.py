"""The serving checkpoint contract.

A checkpoint has to be self-describing and has to fail closed. Every test here
asserts a refusal, not a success: the interesting failures are a checkpoint built
by a different preprocessing version, one whose architecture changed, one whose
heads do not match the run, and a debug-encoder artefact that must never be
served as a result.

The network under test is the debug encoder — a real ``LungSegNet`` with tiny
widths, so this runs on CPU in about a second and never touches ``transformers``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from app.organs.lungs.segmentation.checkpoint import (
    ALLOW_DEBUG_ENV,
    CHECKPOINT_ENV,
    SegmentationUnavailableError,
    build_serving_checkpoint,
    default_checkpoint_path,
    load_serving_checkpoint,
    read_sidecar,
    save_serving_checkpoint,
    sidecar_path_for,
)
from app.organs.lungs.segmentation.model import build_debug_spec, build_model
from app.organs.lungs.segmentation.postprocess import LungPost, OpacityPost
from app.organs.lungs.segmentation.preprocessor import preprocess_spec


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────


def _payload(heads: tuple[str, ...] = ("lung",)) -> dict:
    spec = build_debug_spec(heads)
    model = build_model(spec, pretrained_source=None)
    opacity_post = OpacityPost() if "opacity" in heads else None
    return build_serving_checkpoint(
        model.canonical_state_dict(),
        spec,
        heads_trained=heads,
        postprocess={"lung": LungPost().to_dict(),
                     "opacity": opacity_post.to_dict() if opacity_post else None,
                     "tuned_on_validation": True},
        metrics={"test": {"lung_field": {"dice": {"mean": 0.9, "median": 0.91, "n": 10}}}},
        training={"stage": "lung", "epoch": 3, "weights": "ema", "seed": 42},
    )


@pytest.fixture()
def checkpoint(tmp_path: Path) -> str:
    path = str(tmp_path / "lungs_seg.pth")
    save_serving_checkpoint(_payload(), path)
    return path


@pytest.fixture(autouse=True)
def _allow_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    """The debug encoder is refused unless the process opts in; these tests opt in."""
    monkeypatch.setenv(ALLOW_DEBUG_ENV, "1")


# ─────────────────────────────────────────────────────────────────────────────
# round trip
# ─────────────────────────────────────────────────────────────────────────────


def test_round_trip_carries_everything_inference_needs(checkpoint: str) -> None:
    payload = load_serving_checkpoint(checkpoint)

    assert payload["spec"]["encoder"] == "debug-tiny"
    assert tuple(payload["spec"]["heads"]) == ("lung",)
    assert payload["preprocess"]["version"] == preprocess_spec()["version"]
    assert payload["labels"]["heads_trained"] == ["lung"]
    assert payload["postprocess"]["lung"]["threshold"] == 0.5
    assert payload["postprocess"]["opacity"] is None
    assert payload["metrics"]["test"]["lung_field"]["dice"]["mean"] == 0.9
    assert payload["training"]["seed"] == 42
    # weights travel on CPU in float32 so deployment is a file copy
    first = next(iter(payload["state_dict"].values()))
    assert first.dtype == torch.float32
    assert str(first.device) == "cpu"


def test_weights_actually_rebuild_the_same_network(checkpoint: str) -> None:
    payload = load_serving_checkpoint(checkpoint)
    spec = build_debug_spec(("lung",))
    model = build_model(spec, pretrained_source=None)
    model.load_state_dict(payload["state_dict"], strict=True)

    model.eval()
    with torch.no_grad():
        logits, opacity = model(torch.zeros(1, 3, 256, 256))
    assert logits.shape == (1, 256, 256)
    assert opacity is None          # lung-only stage has no head B


def test_sidecar_is_written_next_and_is_readable(checkpoint: str) -> None:
    sidecar = sidecar_path_for(checkpoint)
    assert os.path.isfile(sidecar)

    with open(sidecar, encoding="utf-8") as handle:
        written = json.load(handle)
    assert written["label_space"]["heads_trained"] == ["lung"]
    assert written["preprocess"]["version"] == preprocess_spec()["version"]

    read = read_sidecar(checkpoint)
    assert read is not None
    assert read["checkpoint"] == "lungs_seg.pth"


# ─────────────────────────────────────────────────────────────────────────────
# refusals
# ─────────────────────────────────────────────────────────────────────────────


def test_missing_checkpoint_is_a_typed_error(tmp_path) -> None:
    with pytest.raises(SegmentationUnavailableError) as caught:
        load_serving_checkpoint(str(tmp_path / "absent.pth"))
    assert caught.value.code == "checkpoint_missing"
    assert "lungs_seg.pth" in str(caught.value)


def test_empty_file_is_refused(tmp_path) -> None:
    path = str(tmp_path / "empty.pth")
    open(path, "wb").close()
    with pytest.raises(SegmentationUnavailableError, match="empty"):
        load_serving_checkpoint(path)


def test_unreadable_file_is_refused_with_a_reason(tmp_path) -> None:
    path = str(tmp_path / "garbage.pth")
    with open(path, "wb") as handle:
        handle.write(b"\x00" * 512)
    with pytest.raises(SegmentationUnavailableError):
        load_serving_checkpoint(path)


def test_preprocessing_version_mismatch_is_refused(tmp_path) -> None:
    """Serving must refuse a contract it does not implement, never resize to fit."""
    payload = _payload()
    payload["preprocess"] = {**payload["preprocess"], "version": 999}
    path = str(tmp_path / "old.pth")
    torch.save(payload, path)

    with pytest.raises(SegmentationUnavailableError, match="preprocess"):
        load_serving_checkpoint(path)


def test_debug_checkpoint_is_refused_without_the_opt_in(
    checkpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ALLOW_DEBUG_ENV, raising=False)
    with pytest.raises(SegmentationUnavailableError, match="debug encoder"):
        load_serving_checkpoint(checkpoint)


def test_heads_trained_must_be_a_subset_of_the_architecture(tmp_path) -> None:
    spec = build_debug_spec(("lung",))
    model = build_model(spec, pretrained_source=None)
    with pytest.raises(ValueError, match="subset"):
        build_serving_checkpoint(
            model.canonical_state_dict(), spec, heads_trained=("lung", "opacity"),
            postprocess={"lung": LungPost().to_dict(), "opacity": None},
            metrics={}, training={},
        )


def test_checkpoint_whose_architecture_lost_the_lung_head_is_refused(tmp_path) -> None:
    payload = _payload()
    payload["spec"]["heads"] = ["opacity"]
    payload["labels"]["heads_trained"] = ["opacity"]
    path = str(tmp_path / "no_lung.pth")
    torch.save(payload, path)

    with pytest.raises(SegmentationUnavailableError):
        load_serving_checkpoint(path)


# ─────────────────────────────────────────────────────────────────────────────
# resolution
# ─────────────────────────────────────────────────────────────────────────────


def test_checkpoint_path_can_be_pointed_elsewhere(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(CHECKPOINT_ENV, str(tmp_path / "custom.pth"))
    assert default_checkpoint_path() == str(tmp_path / "custom.pth")


def test_default_checkpoint_path_is_inside_the_package(monkeypatch) -> None:
    monkeypatch.delenv(CHECKPOINT_ENV, raising=False)
    path = default_checkpoint_path()
    assert path.endswith("lungs_seg.pth")
    assert "lungs/segmentation/checkpoints" in path.replace(os.sep, "/")
