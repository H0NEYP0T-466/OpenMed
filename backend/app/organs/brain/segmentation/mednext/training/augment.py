"""Training-time augmentation for single-channel MRI slices.

Operates on the canvas the preprocessor produces (``uint8`` image, ``uint8``
{0, 1} mask, both ``size x size``) and returns the same, so normalisation —
shared with serving — always runs last on whatever the augmenter produced.

Geometric transforms are applied to image and mask together; masks are warped
as floats and re-thresholded so boundaries stay smooth. Intensity transforms
touch the image only. Every transform is a no-op with the probability left over
from its ``p``, and every random draw comes from the generator passed in, so a
sample is reproducible from ``(seed, epoch, index)``.

Choices made on purpose:
  * horizontal flip only — slices are not guaranteed to be axial, and a vertical
    flip would put the brain upside down relative to anything seen at serving;
  * no cut-out: erasing part of the image while keeping the mask would teach the
    network to hallucinate lesions where there is no evidence;
  * JPEG re-compression and low-resolution round trips, because the training
    data is JPEG/PNG from two sources and serving receives arbitrary uploads.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _affine_matrix(size: int, rng: np.random.Generator, cfg: Any) -> np.ndarray:
    angle = math.radians(rng.uniform(-cfg.aug_rotate_deg, cfg.aug_rotate_deg))
    shear = math.radians(rng.uniform(-cfg.aug_shear_deg, cfg.aug_shear_deg))
    scale = rng.uniform(*cfg.aug_scale_range)
    shift = rng.uniform(-cfg.aug_translate_frac, cfg.aug_translate_frac, size=2) * size

    cos, sin = math.cos(angle), math.sin(angle)
    rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)
    shear_m = np.array([[1.0, math.tan(shear)], [0.0, 1.0]], dtype=np.float64)
    linear = scale * rotation @ shear_m

    centre = np.array([size / 2.0, size / 2.0])
    offset = centre - linear @ centre + shift
    return np.hstack([linear, offset[:, None]]).astype(np.float32)


def _warp(image: np.ndarray, mask: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    size = image.shape[0]
    warped = cv2.warpAffine(
        image, matrix, (size, size), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    warped_mask = cv2.warpAffine(
        mask.astype(np.float32), matrix, (size, size), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return warped, (warped_mask >= 0.5).astype(np.uint8)


def _elastic(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    size = image.shape[0]
    magnitude = rng.uniform(1.5, 4.0) * size / 256.0
    sigma = rng.uniform(8.0, 14.0) * size / 256.0

    def field() -> np.ndarray:
        smooth = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), sigma)
        return smooth / (float(smooth.std()) + 1e-6) * magnitude

    dx, dy = field(), field()
    grid_x, grid_y = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32))
    map_x, map_y = grid_x + dx, grid_y + dy
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    warped_mask = cv2.remap(
        mask.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    return warped, (warped_mask >= 0.5).astype(np.uint8)


def _bias_field(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size = image.shape[0]
    amplitude = rng.uniform(0.1, 0.3)
    coarse = rng.uniform(-amplitude, amplitude, (4, 4)).astype(np.float32)
    field = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
    return image * (1.0 + field)


def _jpeg(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    quality = int(rng.integers(55, 96))
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return image
    decoded = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    return image if decoded is None else decoded


def _intensity(image: np.ndarray, rng: np.random.Generator, cfg: Any) -> np.ndarray:
    out = image.astype(np.float32)

    if rng.random() < cfg.aug_gamma_p:
        gamma = rng.uniform(*cfg.aug_gamma_range)
        out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, gamma)
    if rng.random() < cfg.aug_bias_field_p:
        out = _bias_field(out, rng)
    if rng.random() < cfg.aug_blur_p:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 1.3))
    if rng.random() < cfg.aug_lowres_p:
        size = out.shape[0]
        factor = rng.uniform(0.5, 0.8)
        small = cv2.resize(out, (int(size * factor),) * 2, interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (size, size), interpolation=cv2.INTER_LINEAR)
    if rng.random() < cfg.aug_noise_p:
        out = out + rng.normal(0.0, rng.uniform(1.0, 6.0), out.shape).astype(np.float32)

    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < cfg.aug_jpeg_p:
        out = _jpeg(out, rng)
    return out


def augment(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, cfg: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Return an augmented ``(image, mask)`` pair, keeping the lesion in frame."""
    had_lesion = bool(mask.any())
    img, msk = image, mask

    if rng.random() < cfg.aug_hflip_p:
        img, msk = img[:, ::-1], msk[:, ::-1]
    img, msk = np.ascontiguousarray(img), np.ascontiguousarray(msk)
    flipped = (img, msk)

    if rng.random() < cfg.aug_affine_p:
        img, msk = _warp(img, msk, _affine_matrix(img.shape[0], rng, cfg))
    if rng.random() < cfg.aug_elastic_p:
        img, msk = _elastic(img, msk, rng)

    if had_lesion and not msk.any():
        img, msk = flipped

    img = _intensity(img, rng, cfg)
    return img, msk
