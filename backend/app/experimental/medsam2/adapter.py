"""MedSAM2 — inference only, box-prompted, lazily loaded.

MedSAM2 is the medical fine-tune of SAM2. It needs SAM2's *architecture*, but
the weights are MedSAM2's — the fork vendored under `_vendor/MedSAM2` is the one
the checkpoint was trained against, and the vanilla PyPI `sam2` package is not a
substitute: same class names, different state-dict keys.

Nothing here trains. The model is loaded on first use, kept while its tab is
open, and released on unload so the lab does not hold 39 M parameters for a
model nobody is looking at.
"""

from __future__ import annotations

import base64
import io
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from ..base import LabAdapter, MaskResult, SegPrompt, resolve_device
from . import weights

logger = logging.getLogger(__name__)

VENDOR_DIR = Path(__file__).resolve().parent / "_vendor" / "MedSAM2"
CONFIG_DIR_RELPATH = "sam2/configs"
CONFIG_NAME = "sam2.1_hiera_t512.yaml"


def _encode_png(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class MedSAM2Adapter(LabAdapter):
    id = "medsam2"
    title = "MedSAM2"
    description = "Promptable medical segmentation · box prompt · SAM2.1 Hiera-T @ 512"
    input_size = 512

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._predictor: Optional[Any] = None
        self._device: str = "cpu"
        self._load_ms: Optional[float] = None

    # ── availability ───────────────────────────────────────────────────────

    def availability(self) -> tuple[bool, str]:
        if not VENDOR_DIR.is_dir():
            return False, (
                f"MedSAM2 source missing at {VENDOR_DIR}. "
                "Clone https://github.com/bowang-lab/MedSAM2 into that folder."
            )
        return weights.status()

    # ── lifecycle ──────────────────────────────────────────────────────────

    def load(self) -> float:
        if self._predictor is not None:
            return 0.0
        started = time.perf_counter()

        checkpoint = weights.ensure()
        vendor = str(VENDOR_DIR)
        if vendor not in sys.path:
            sys.path.insert(0, vendor)

        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._device = resolve_device()
        config_dir = str(VENDOR_DIR / CONFIG_DIR_RELPATH)
        # hydra resolves config *names* against an initialised directory; a bare
        # filesystem path is read as a name and never found.
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            model = build_sam2(CONFIG_NAME, str(checkpoint), device=self._device)
        model.eval()
        self._model = model
        self._predictor = SAM2ImagePredictor(model)
        self._load_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "MedSAM2 loaded in %.0f ms on %s (%.1fM params)",
            self._load_ms,
            self._device,
            sum(p.numel() for p in model.parameters()) / 1e6,
        )
        return self._load_ms

    def unload(self) -> None:
        self._predictor = None
        self._model = None
        self._load_ms = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("MedSAM2 unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    @staticmethod
    def _mask_logits(mask_png: bytes, target_hw: tuple[int, int]) -> np.ndarray:
        """Binary mask PNG → logits on the decoder's own grid.

        SAM2 does not resize a mask prompt for you. The prompt encoder runs the
        mask through a stride-4 downsampling stack and adds the result to the
        image embedding, so the input must be *four times* the embedding's
        spatial size: 32×32 embedding → 128×128 mask. 256 (SAM1's convention)
        or 32 (the embedding itself) both fail with a shape mismatch.

        A binary mask becomes ±10 logits: a strong prior that a few correction
        clicks can still move.
        """
        with Image.open(io.BytesIO(mask_png)) as handle:
            mask = handle.convert("L").resize((target_hw[1], target_hw[0]), Image.BILINEAR)
        array = np.asarray(mask, dtype=np.float32) / 255.0
        return ((array * 2.0) - 1.0)[None, None, :, :] * 10.0

    # ── inference ──────────────────────────────────────────────────────────

    def predict(self, image: Image.Image, prompt: SegPrompt) -> MaskResult:
        if self._predictor is None:
            self.load()
        assert self._predictor is not None

        started = time.perf_counter()
        rgb = image.convert("RGB")
        width, height = rgb.size

        box = prompt.clamped_box(width, height)
        points = prompt.clamped_points(width, height)

        point_coords = None
        point_labels = None
        if points:
            point_coords = np.array([[p.x, p.y] for p in points], dtype=np.float32)
            point_labels = np.array([1 if p.positive else 0 for p in points], dtype=np.int32)

        array = np.asarray(rgb, dtype=np.uint8)
        self._predictor.set_image(array)

        mask_input = None
        if prompt.mask_png is not None:
            # The mask prompt must match the image embedding's grid, which only
            # exists after set_image. Private attribute: SAM2 exposes no getter.
            embed = self._predictor._features["image_embed"]
            mask_input = self._mask_logits(
                prompt.mask_png, (int(embed.shape[-2]) * 4, int(embed.shape[-1]) * 4)
            )

        masks, scores, _logits = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=np.array([box], dtype=np.float32) if box else None,
            mask_input=mask_input,
            multimask_output=False,
        )

        mask = np.asarray(masks[0], dtype=bool)
        if mask.ndim == 3:
            mask = mask[0]
        score = float(np.asarray(scores).reshape(-1)[0])

        mask_uint8 = (mask.astype(np.uint8)) * 255
        overlay = array.copy()
        tint = np.zeros_like(overlay)
        tint[..., 0] = 255
        alpha = mask[..., None].astype(np.float32) * 0.42
        overlay = (overlay.astype(np.float32) * (1 - alpha) + tint.astype(np.float32) * alpha).astype(np.uint8)

        return MaskResult(
            mask_base64=_encode_png(mask_uint8),
            overlay_base64=_encode_png(overlay),
            score=score,
            inference_ms=(time.perf_counter() - started) * 1000.0,
            image_size=(width, height),
            model=self.id,
            extra={
                "prompt_kinds": prompt.kinds,
                "prompt_box": list(box) if box else None,
                "prompt_points": [[p.x, p.y, 1 if p.positive else 0] for p in points],
                "mask_pixels": int(mask.sum()),
                "mask_fraction": float(mask.mean()),
                "device": self._device,
                "load_ms": self._load_ms,
            },
        )
