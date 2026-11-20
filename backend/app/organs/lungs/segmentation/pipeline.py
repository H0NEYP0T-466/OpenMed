"""Lungs segmentation serving pipeline.

Two passes, by design:

1. lung field on the full 256 x 256 frame;
2. COVID-type opacity on a crop around the predicted lung field, resized back to
   256 x 256, then pasted back and clipped to the lung field.

The pipeline is also what the training package uses for the final held-out test,
so the number reported is the number the application produces: decode the
original file, run inference, resample back, compare with the original-resolution
mask. Nothing here imports another organ, and nothing is constructed at import
time (no weights, no ``transformers``).
"""

from __future__ import annotations

import base64
import contextlib
import ctypes
import gc
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import torch

from .checkpoint import (
    CHECKPOINT_NAME,
    SegmentationUnavailableError,
    default_checkpoint_path,
    load_serving_checkpoint,
    read_sidecar,
)
from .labels import SCOPE_NOTICE, TASK_NAME, merge_label_map
from .model import MODEL_TAG, SegSpec, build_model, count_parameters
from .postprocess import (
    LungPost,
    OpacityPost,
    component_areas,
    postprocess_lung,
    postprocess_opacity,
)
from .preprocessor import (
    CROP_MARGIN,
    PREPROCESS_VERSION,
    Geometry,
    canvas_to_original,
    canvases_to_tensor,
    crop_resize_image,
    decode_gray,
    full_box,
    lung_crop_box,
    to_canvas,
    to_gray_uint8,
    uncrop_plane,
)

logger = logging.getLogger(__name__)

OUTPUT_MAX_SIDE = 768

__all__ = [
    "CHECKPOINT_NAME",
    "LungSegmenter",
    "SegResult",
    "SegmentationUnavailableError",
    "default_checkpoint_path",
    "encode_label_map_png",
    "encode_mask_png",
    "info_from_payload",
    "release_runtime_memory",
]


def release_runtime_memory() -> None:
    """Hand the weights' memory back, not just drop the attribute."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if sys.platform.startswith("linux"):
        with contextlib.suppress(OSError, AttributeError):  # best effort on non-glibc systems
            ctypes.CDLL("libc.so.6").malloc_trim(0)


# ------------------------------------------------------------------------ PNG output


def _fit(image: np.ndarray, max_side: int, interpolation: int) -> np.ndarray:
    longest = max(image.shape[:2])
    if longest <= max_side:
        return image
    scale = max_side / longest
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=interpolation)


def _png_data_url(plane: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", plane)
    if not ok:
        raise RuntimeError("PNG encoding failed for a segmentation output.")
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def encode_mask_png(mask: np.ndarray, max_side: int = OUTPUT_MAX_SIDE) -> str:
    """Binary mask as white-on-black PNG (255 / 0)."""
    plane = (mask > 0).astype(np.uint8) * 255
    plane = _fit(plane, max_side, cv2.INTER_LINEAR)
    return _png_data_url((plane > 127).astype(np.uint8) * 255)


def encode_label_map_png(label_map: np.ndarray, max_side: int = OUTPUT_MAX_SIDE) -> str:
    """Label map whose pixel *values* are 0 / 1 / 2. Not for display."""
    plane = _fit(label_map.astype(np.uint8), max_side, cv2.INTER_NEAREST)
    return _png_data_url(plane)


def output_size(shape: tuple[int, int], max_side: int = OUTPUT_MAX_SIDE) -> tuple[int, int]:
    """``(height, width)`` of an encoded output for a mask of ``shape``."""
    longest = max(shape)
    if longest <= max_side:
        return int(shape[0]), int(shape[1])
    scale = max_side / longest
    return max(1, round(shape[0] * scale)), max(1, round(shape[1] * scale))


# ------------------------------------------------------------------------- results


@dataclass
class SegResult:
    lung_mask: np.ndarray
    opacity_mask: np.ndarray | None
    lung_probability: np.ndarray
    opacity_probability: np.ndarray | None
    label_map: np.ndarray
    original_size: tuple[int, int]
    crop_box: list[int] | None
    crop_fallback: bool
    opacity_available: bool
    opacity_unavailable_reason: str | None
    lung_post: LungPost
    opacity_post: OpacityPost | None
    inference_ms: float
    stats: dict[str, Any] = field(default_factory=dict)


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _stats(
    lung: np.ndarray,
    opacity: np.ndarray | None,
    opacity_prob: np.ndarray | None,
    crop_box: list[int] | None,
    crop_fallback: bool,
) -> dict[str, Any]:
    lung_area = int(lung.sum())
    stats: dict[str, Any] = {
        "lung_area_px": lung_area,
        "lung_fraction_of_frame": lung_area / lung.size,
        "lung_components": component_areas(lung)[0],
        "lung_bbox": _bbox(lung),
        "opacity_area_px": None,
        "opacity_fraction_of_lung": None,
        "opacity_components": None,
        "opacity_peak_probability": None,
        "opacity_mean_probability": None,
        "crop_box": crop_box,
        "crop_fallback": crop_fallback,
    }
    if opacity is not None:
        area = int(opacity.sum())
        stats["opacity_area_px"] = area
        stats["opacity_fraction_of_lung"] = (area / lung_area) if lung_area else 0.0
        stats["opacity_components"] = component_areas(opacity)[0]
        if opacity_prob is not None and lung_area:
            inside = opacity_prob[lung > 0]
            stats["opacity_peak_probability"] = float(inside.max()) if inside.size else 0.0
        if opacity_prob is not None and area:
            stats["opacity_mean_probability"] = float(opacity_prob[opacity > 0].mean())
        elif area == 0:
            stats["opacity_mean_probability"] = 0.0
    return stats


# ------------------------------------------------------------------ checkpoint info


def _file_facts(path: str) -> dict[str, Any]:
    try:
        stat = os.stat(path)
    except OSError:
        return {"name": os.path.basename(path), "size_mb": None, "modified": None}
    return {
        "name": os.path.basename(path),
        "size_mb": round(stat.st_size / 2**20, 2),
        "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
    }


def _pick_metrics(payload: dict[str, Any], sidecar: dict[str, Any] | None) -> dict[str, Any]:
    source = (sidecar or {}).get("metrics") or payload.get("metrics") or {}
    return {"test": source.get("test"), "val": source.get("val")}


def info_from_payload(
    payload: dict[str, Any],
    path: str,
    *,
    loaded: bool,
    device: str | None,
    parameters: int | None = None,
) -> dict[str, Any]:
    """The ``/seg-model-info`` body, built from a validated checkpoint payload."""
    sidecar = read_sidecar(path)
    if parameters is None:
        parameters = int(sum(t.numel() for t in payload["state_dict"].values()))
    training = payload.get("training", {})
    labels = payload["labels"]
    size = int(payload["preprocess"]["image_size"])
    return {
        "task": labels.get("task", TASK_NAME),
        "model_tag": MODEL_TAG,
        "parameters": parameters,
        "input_size": f"{size}x{size}",
        "loaded": loaded,
        "device": device,
        "heads": list(payload["spec"]["heads"]),
        "heads_trained": list(labels["heads_trained"]),
        "preprocess_version": int(payload["preprocess"]["version"]),
        "postprocess": payload["postprocess"],
        "metrics": _pick_metrics(payload, sidecar),
        "training": {
            "epoch": training.get("epoch"),
            "weights": training.get("weights"),
            "stage": training.get("stage"),
            "seed": training.get("seed"),
            "split_counts": training.get("split_counts"),
        },
        "checkpoint": _file_facts(path),
        "scope_notice": SCOPE_NOTICE,
    }


# -------------------------------------------------------------------------- network


class LungSegmenter:
    """Loads a serving checkpoint and segments one image at a time."""

    def __init__(self, model_path: str | None = None, device: str | None = None) -> None:
        self.model_path = model_path or default_checkpoint_path()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: torch.nn.Module | None = None
        self._lock = threading.Lock()
        self._payload_meta: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        started = time.perf_counter()
        payload = load_serving_checkpoint(self.model_path)
        try:
            self.spec = SegSpec.from_dict(payload["spec"])
            model = build_model(self.spec, pretrained_source=None)
            model.load_canonical(payload["state_dict"])
        except Exception as exc:  # noqa: BLE001 - mismatched weights must fail closed
            raise SegmentationUnavailableError(
                f"Lungs segmentation weights do not match the architecture they declare: {exc}"
            ) from exc

        self.model = model.to(self.device).eval()
        self.size = int(payload["preprocess"]["image_size"])
        self.heads_trained = tuple(payload["labels"]["heads_trained"])
        post = payload["postprocess"]
        self.lung_post = LungPost.from_dict(post.get("lung", {}))
        self.opacity_post = (
            OpacityPost.from_dict(post["opacity"]) if post.get("opacity") else None
        )
        self.parameters = count_parameters(model)
        self.load_ms = (time.perf_counter() - started) * 1000.0
        self._payload_meta = info_from_payload(
            payload, self.model_path, loaded=True, device=str(self.device),
            parameters=self.parameters,
        )
        logger.info(
            "Lungs segmenter loaded in %.0f ms on %s: %.2fM params, heads %s (trained %s), "
            "preprocess v%d",
            self.load_ms, self.device, self.parameters / 1e6, list(self.spec.heads),
            list(self.heads_trained), PREPROCESS_VERSION,
        )

    # ---------------------------------------------------------------- lifecycle

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def unload(self) -> None:
        model = self.model
        self.model = None
        del model
        release_runtime_memory()

    # --------------------------------------------------------------- capability

    @property
    def opacity_available(self) -> bool:
        return (
            self.model is not None
            and getattr(self.model, "has_opacity", False)
            and "opacity" in self.heads_trained
            and self.opacity_post is not None
        )

    @property
    def opacity_unavailable_reason(self) -> str | None:
        if self.opacity_available:
            return None
        if "opacity" not in self.spec.heads:
            return (
                "This checkpoint has no opacity head (trained with --stage lung); only the lung "
                "field is segmented."
            )
        if "opacity" not in self.heads_trained:
            return "The opacity head in this checkpoint was not trained; only the lung field is segmented."
        return "Opacity post-processing was not recorded in this checkpoint; only the lung field is segmented."

    # ---------------------------------------------------------------- inference

    @torch.no_grad()
    def _probabilities(self, canvases: list[np.ndarray], head: str) -> np.ndarray:
        if self.model is None:
            raise SegmentationUnavailableError("Lungs segmentation model is not loaded.")
        batch = canvases_to_tensor(canvases).to(self.device)
        use_amp = self.device.type == "cuda"
        with self._lock, torch.autocast(self.device.type, dtype=torch.float16, enabled=use_amp):
            logits = (
                self.model.lung_logits(batch) if head == "lung" else self.model.opacity_logits(batch)
            )
        return torch.sigmoid(logits.float()).cpu().numpy()

    def canvas_probability(self, canvas: np.ndarray, head: str, tta: bool) -> np.ndarray:
        """Probability on a 256 x 256 canvas. Flip-TTA is an involution; no label swap."""
        variants = [canvas]
        if tta:
            variants.append(np.ascontiguousarray(canvas[:, ::-1]))
        probs = self._probabilities(variants, head)
        merged = probs[0]
        if tta:
            merged = 0.5 * (merged + probs[1][:, ::-1])
        return np.ascontiguousarray(merged, dtype=np.float32)

    def predict_gray(
        self,
        gray: np.ndarray,
        *,
        lung_post: LungPost | None = None,
        opacity_post: OpacityPost | None = None,
        run_opacity: bool = True,
    ) -> SegResult:
        started = time.perf_counter()
        gray = to_gray_uint8(gray)
        lung_post = lung_post or self.lung_post
        canvas, geom = to_canvas(gray, self.size)

        lung_canvas_prob = self.canvas_probability(canvas, "lung", lung_post.tta_hflip)
        lung_canvas_mask = postprocess_lung(lung_canvas_prob, lung_post)
        lung_prob = canvas_to_original(lung_canvas_prob, geom).astype(np.float32)
        lung_mask = lung_canvas_mask if geom.is_native else postprocess_lung(lung_prob, lung_post)

        opacity_mask: np.ndarray | None = None
        opacity_prob: np.ndarray | None = None
        crop_box: list[int] | None = None
        crop_fallback = False
        used_opacity_post: OpacityPost | None = None

        if run_opacity and self.opacity_available:
            used_opacity_post = opacity_post or self.opacity_post
            assert used_opacity_post is not None
            box = lung_crop_box(lung_canvas_mask, CROP_MARGIN)
            if box is None:
                box, crop_fallback = full_box(canvas.shape), True
            crop_box = [int(v) for v in box]
            crop = crop_resize_image(canvas, box, self.size)
            crop_prob = self.canvas_probability(crop, "opacity", used_opacity_post.tta_hflip)
            canvas_prob = uncrop_plane(crop_prob, box, canvas.shape)
            opacity_prob = canvas_to_original(canvas_prob, geom).astype(np.float32)
            opacity_mask = postprocess_opacity(opacity_prob, lung_mask, used_opacity_post)

        label_map = merge_label_map(lung_mask, opacity_mask)
        return SegResult(
            lung_mask=lung_mask,
            opacity_mask=opacity_mask,
            lung_probability=lung_prob,
            opacity_probability=opacity_prob,
            label_map=label_map,
            original_size=(geom.original_h, geom.original_w),
            crop_box=crop_box,
            crop_fallback=crop_fallback,
            opacity_available=self.opacity_available,
            opacity_unavailable_reason=self.opacity_unavailable_reason,
            lung_post=lung_post,
            opacity_post=used_opacity_post,
            inference_ms=(time.perf_counter() - started) * 1000.0,
            stats=_stats(lung_mask, opacity_mask, opacity_prob, crop_box, crop_fallback),
        )

    def predict_bytes(self, data: bytes, **kwargs: Any) -> tuple[SegResult, np.ndarray]:
        """Decode with the training-time decoder, then segment. Returns ``(result, gray)``."""
        gray = decode_gray(data)
        return self.predict_gray(gray, **kwargs), gray

    # ----------------------------------------------------------------- metadata

    def model_info(self) -> dict[str, Any]:
        info = dict(self._payload_meta)
        info["loaded"] = self.is_loaded
        info["device"] = str(self.device)
        return info

    def geometry_for(self, gray: np.ndarray) -> Geometry:
        return Geometry(gray.shape[0], gray.shape[1], self.size)
