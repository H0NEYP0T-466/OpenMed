"""Training-time augmentation.

Geometric transforms are applied to the image and every mask together (one matrix);
intensity transforms touch the image only, on the uint8 canvas *before* the shared
preprocessing contract turns it into a tensor. Nothing here exists at inference.

Deliberately absent: cutout / random erasing. Erasing the image while keeping the
mask teaches the network to hallucinate structure that is not there.

Horizontal flip is safe because head A is bilateral and head B is not lateralised;
at serving the same flip is used as TTA and is an involution (no label swap).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..preprocessor import CROP_MARGIN, CROP_MIN_SIDE, full_box, grow_box
from .config import TrainConfig


def _p(base: float, strength: float) -> float:
    return float(np.clip(base * strength, 0.0, 1.0))


def affine_matrix(
    rng: np.random.Generator, size: int, cfg: TrainConfig, strength: float
) -> np.ndarray:
    angle = rng.uniform(-cfg.aug_rotate_deg, cfg.aug_rotate_deg) * strength
    lo, hi = cfg.aug_scale_range
    scale = 1.0 + (rng.uniform(lo, hi) - 1.0) * strength
    shift = cfg.aug_translate_frac * size * strength
    tx, ty = rng.uniform(-shift, shift), rng.uniform(-shift, shift)
    centre = (size / 2.0, size / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, scale)
    matrix[:, 2] += (tx, ty)
    return matrix


def geometric(
    image: np.ndarray,
    masks: list[np.ndarray | None],
    rng: np.random.Generator,
    cfg: TrainConfig,
) -> tuple[np.ndarray, list[np.ndarray | None]]:
    strength = cfg.aug_strength
    if rng.random() < cfg.aug_hflip_p:
        image = np.ascontiguousarray(image[:, ::-1])
        masks = [None if m is None else np.ascontiguousarray(m[:, ::-1]) for m in masks]
    if rng.random() < _p(cfg.aug_affine_p, strength):
        size = image.shape[0]
        matrix = affine_matrix(rng, size, cfg, strength)
        image = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped: list[np.ndarray | None] = []
        for m in masks:
            if m is None:
                warped.append(None)
                continue
            plane = cv2.warpAffine(m.astype(np.float32), matrix, (size, size),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
            warped.append((plane >= 0.5).astype(np.uint8))
        masks = warped
    return image, masks


def intensity(image: np.ndarray, rng: np.random.Generator, cfg: TrainConfig) -> np.ndarray:
    strength = cfg.aug_strength
    out = image
    if rng.random() < _p(cfg.aug_contrast_p, strength):
        contrast = 1.0 + rng.uniform(-cfg.aug_contrast, cfg.aug_contrast) * strength
        brightness = rng.uniform(-cfg.aug_brightness, cfg.aug_brightness) * strength * 255.0
        out = np.clip(out.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
    if rng.random() < _p(cfg.aug_gamma_p, strength):
        lo, hi = cfg.aug_gamma_range
        gamma = float(np.exp(np.log(rng.uniform(lo, hi)) * strength))
        out = (np.power(out.astype(np.float32) / 255.0, gamma) * 255.0).clip(0, 255).astype(np.uint8)
    if rng.random() < _p(cfg.aug_lowres_p, strength):
        lo, hi = cfg.aug_lowres_scale
        factor = float(rng.uniform(lo, hi))
        side = out.shape[0]
        small = cv2.resize(out, (max(8, int(side * factor)),) * 2, interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (side, side), interpolation=cv2.INTER_LINEAR)
    if rng.random() < _p(cfg.aug_jpeg_p, strength):
        quality = int(rng.integers(cfg.aug_jpeg_quality[0], cfg.aug_jpeg_quality[1] + 1))
        ok, buffer = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            decoded = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
            if decoded is not None:
                out = decoded
    return out


def augment(
    image: np.ndarray,
    lung: np.ndarray,
    opacity: np.ndarray | None,
    rng: np.random.Generator,
    cfg: TrainConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    image, (lung_a, opacity_a) = geometric(image, [lung, opacity], rng, cfg)
    assert lung_a is not None
    return intensity(image, rng, cfg), lung_a, opacity_a


def jittered_crop_box(
    lung_mask: np.ndarray, rng: np.random.Generator, cfg: TrainConfig
) -> tuple[int, int, int, int]:
    """Lung-bbox crop with a randomised margin and edges, so head B tolerates a
    predicted (imperfect) lung mask at serving. Falls back to the full frame."""
    ys, xs = np.nonzero(lung_mask)
    if xs.size == 0:
        return full_box(lung_mask.shape)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    width, height = x1 - x0, y1 - y0
    jitter = cfg.crop_jitter * cfg.aug_strength
    x0 += round(rng.uniform(-jitter, jitter) * width)
    x1 += round(rng.uniform(-jitter, jitter) * width)
    y0 += round(rng.uniform(-jitter, jitter) * height)
    y1 += round(rng.uniform(-jitter, jitter) * height)
    if x1 <= x0 or y1 <= y0:
        return full_box(lung_mask.shape)
    margin = max(0.0, CROP_MARGIN + rng.uniform(-cfg.crop_margin_jitter, cfg.crop_margin_jitter))
    return grow_box((x0, y0, x1, y1), lung_mask.shape, margin, CROP_MIN_SIDE)
