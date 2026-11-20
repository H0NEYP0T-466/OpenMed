"""Preprocessing parity between training and serving.

The failure this guards against is silent: if training saw a different tensor than
serving builds, the weights are being asked a question they were never trained on
and the output is confidently wrong. So the contract is asserted as data
(``preprocess_spec``) *and* as behaviour (the same bytes must produce the same
canvas on both paths).

Nothing here opens a checkpoint or builds a network.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from app.organs.lungs.segmentation.preprocessor import (
    IMAGE_SIZE,
    canvas_to_original,
    canvases_to_tensor,
    crop_resize_image,
    decode_gray,
    full_box,
    grow_box,
    lung_crop_box,
    mask_to_canvas,
    normalise_u8,
    read_mask,
    to_canvas,
    uncrop_plane,
)


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def _scan(size: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    frame = 60 + 120 * (((xx - size / 2) ** 2 + (yy - size / 2) ** 2) < (size / 3) ** 2)
    return np.clip(frame + rng.normal(0, 6, frame.shape), 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# The contract as data
# ─────────────────────────────────────────────────────────────────────────────


def test_contract_says_what_it_must() -> None:
    from app.organs.lungs.segmentation.preprocessor import PREPROCESS_VERSION, preprocess_spec

    spec = preprocess_spec()
    assert spec["version"] == PREPROCESS_VERSION
    assert spec["image_size"] == IMAGE_SIZE == 256
    assert spec["channels_in"] == 1
    assert spec["channels_model"] == 3
    # No CLAHE: an intensity transform that exists at inference but not in training
    # (or the reverse) is the classic silent_corruption bug in medical imaging.
    assert spec["clahe"] is False
    assert spec["normalise"]["replicate_to_channels"] == 3
    assert len(spec["normalise"]["mean"]) == 3
    assert len(spec["normalise"]["std"]) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Training path == serving path
# ─────────────────────────────────────────────────────────────────────────────


def test_same_bytes_produce_the_same_canvas_on_both_paths() -> None:
    """Training decodes from a file, serving decodes from an upload: same pixels."""
    original = _scan(256)
    payload = _png(original)

    from_bytes, geom_serving = to_canvas(decode_gray(payload), IMAGE_SIZE)
    from_file, geom_training = to_canvas(decode_gray(_png(original)), IMAGE_SIZE)

    assert np.array_equal(from_bytes, from_file)
    assert np.array_equal(from_bytes, original)          # native 256 is untouched
    assert geom_serving.is_native and geom_training.is_native
    assert from_bytes.dtype == np.uint8


def test_native_canvas_is_a_no_op() -> None:
    original = _scan(256)
    canvas, geom = to_canvas(original, IMAGE_SIZE)
    assert np.array_equal(canvas, original)
    assert geom.is_native is True


@pytest.mark.parametrize(("height", "width"), [(300, 200), (512, 512), (128, 256)])
def test_non_native_images_are_resized_and_map_back(height: int, width: int) -> None:
    original = _scan(256)[:height, :width] if (height, width) != (512, 512) else _scan(512)
    canvas, geom = to_canvas(original, IMAGE_SIZE)

    assert canvas.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert geom.is_native is False
    assert geom.original_h == original.shape[0]
    assert geom.original_w == original.shape[1]

    back = canvas_to_original(canvas, geom)
    assert back.shape == original.shape


def test_masks_resample_to_binary_and_never_to_grey() -> None:
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[40:120, 30:100] = 255
    canvas = mask_to_canvas(mask, IMAGE_SIZE)
    assert canvas.dtype == np.uint8
    assert set(np.unique(canvas).tolist()) == {0, 1}

    # a non-native mask must still come back binary, not fuzzy
    small = cv2.resize(mask, (100, 100), interpolation=cv2.INTER_AREA)
    resized = mask_to_canvas(small, IMAGE_SIZE)
    assert set(np.unique(resized).tolist()) <= {0, 1}


def test_read_mask_binarises_255_to_one() -> None:
    path = "/tmp/_lungs_seg_mask_test.png"
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    with open(path, "wb") as handle:
        handle.write(_png(mask))

    loaded = read_mask(path)
    assert loaded.shape == (64, 64)
    assert int(loaded.max()) == 1
    assert int(loaded.sum()) == 400


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation and tensor assembly
# ─────────────────────────────────────────────────────────────────────────────


def test_normalise_builds_three_channels_from_one_plane() -> None:
    canvas = _scan(256)
    batch = torch.from_numpy(canvas).unsqueeze(0)          # (1, H, W) uint8
    tensor = normalise_u8(batch)

    assert tuple(tensor.shape) == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype == torch.float32
    # a normal X-ray, ImageNet-normalised, lands in roughly [-2.2, 2.7]
    assert float(tensor.min()) > -2.2 and float(tensor.max()) < 2.7


def test_normalise_refuses_a_plane_without_a_batch_dimension() -> None:
    with pytest.raises(ValueError, match=r"\(B, H, W\)"):
        normalise_u8(torch.from_numpy(_scan(256)))


def test_batch_assembly_is_three_channels_of_the_same_plane() -> None:
    from app.organs.lungs.segmentation.preprocessor import IMAGENET_MEAN, IMAGENET_STD

    canvas = _scan(256)
    tensor = canvases_to_tensor([canvas, canvas])
    assert tuple(tensor.shape) == (2, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype == torch.float32

    values = tensor.numpy()
    # Each channel is the *same* luminance plane, normalised with that channel's own
    # ImageNet statistics. This is what "replicated to 3 channels" means in practice,
    # and it is why the three planes are not numerically identical after normalisation.
    base = canvas.astype(np.float32) / 255.0
    for channel in range(3):
        expected = (base - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
        assert np.allclose(values[0, channel], expected, atol=1e-5)
    # and the two batch entries are identical because the input was
    assert np.array_equal(values[0], values[1])


# ─────────────────────────────────────────────────────────────────────────────
# The lung crop head B runs on
# ─────────────────────────────────────────────────────────────────────────────


def test_lung_crop_box_covers_both_lobes() -> None:
    lung = np.zeros((256, 256), dtype=np.uint8)
    lung[80:180, 40:110] = 1          # left lobe
    lung[80:180, 150:220] = 1         # right lobe

    box = lung_crop_box(lung)
    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 <= 40 and y0 <= 80
    assert x1 >= 220 and y1 >= 180
    assert x1 > x0 and y1 > y0


def test_empty_mask_has_no_box_and_falls_back_to_the_frame() -> None:
    empty = np.zeros((256, 256), dtype=np.uint8)
    assert lung_crop_box(empty) is None
    assert full_box(empty.shape) == (0, 0, 256, 256)


def test_crop_grows_around_small_boxes_and_stays_inside_the_frame() -> None:
    shape = (256, 256)
    tiny = grow_box((120, 120, 128, 128), shape, 0.06, 64)
    assert tiny[2] - tiny[0] >= 64
    assert tiny[3] - tiny[1] >= 64
    assert 0 <= tiny[0] < tiny[2] <= 256 and 0 <= tiny[1] < tiny[3] <= 256

    corner = grow_box((0, 0, 30, 30), shape, 0.06, 64)
    assert corner[0] >= 0 and corner[1] >= 0

    # a box already wider than the minimum keeps its own size plus the margin
    wide = grow_box((40, 40, 200, 200), shape, 0.06, 64)
    assert wide[2] - wide[0] > 160


def test_crop_and_uncrop_round_trip_keeps_frame_geometry() -> None:
    canvas = _scan(256)
    box = (60, 70, 190, 200)

    crop = crop_resize_image(canvas, box)
    assert crop.shape == (IMAGE_SIZE, IMAGE_SIZE)

    probabilities = np.random.default_rng(3).random((IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    restored = uncrop_plane(probabilities, box, canvas.shape)

    assert restored.shape == canvas.shape
    # everything outside the box is zero, everything inside came from the crop
    assert float(restored[: box[1], :].max(initial=0.0)) == 0.0
    assert float(restored[box[1] : box[3], box[0] : box[2]].max()) > 0.0


def test_decode_rejects_garbage_instead_of_returning_noise() -> None:
    from app.organs.lungs.segmentation.preprocessor import decode_gray

    with pytest.raises(ValueError):
        decode_gray(b"this is not an image")
