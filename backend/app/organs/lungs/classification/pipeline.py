"""Serving pipeline for the two lung models: LungV2S and stock XRV DenseNet-121.

Model A — LungV2S
    EfficientNetV2-S, 4-class single-label softmax over
    ``COVID / Lung_Opacity / Normal / Viral Pneumonia``.
    Preprocessing: grayscale replicated to 3 channels, resized to the sidecar's
    ``infer_size`` (300 or 384), ImageNet mean/std normalisation. Owned by
    ``model_v2s.preprocess_v2s``.

Model B — XRV stock
    TorchXRayVision ``densenet121-res224-all``, 18-class multilabel sigmoid over
    ``model.pathologies``. Preprocessing: one grayscale channel, resized to 224,
    scaled to [-1024, 1024]. Owned by ``XRVPipeline._preprocess`` below.

The two preprocessing paths deliberately share no code. They are not variants
of one recipe; they are two different input distributions. A LungV2S tensor is
ImageNet-normalised RGB in roughly [-2.1, 2.6] and an XRV tensor is single
channel in [-1024, 1024]. Handing either model the other's tensor raises no
error — both networks accept any float tensor and return a full, confident,
completely meaningless output. The XRV network sees an ImageNet-normalised
image as flat near-zero grey; LungV2S sees a [-1024, 1024] scan as saturated
black. Neither complains. That is why the two normalisations live in different
modules, behind different function names, with no shared helper to accidentally
route one into the other.

Everything here is import-safe: no torch model is constructed and
``torchxrayvision`` is not imported until ``load()`` is called.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from . import model as xrv_model
from .model_v2s import (
    CHECKPOINT_NAME,
    CLASS_NAMES,
    DEFAULT_INFER_SIZE,
    MODEL_TAG as V2S_MODEL_TAG,
    GradCAMV2S,
    build_model as build_v2s_model,
    cam_to_bbox,
    load_sidecar,
    overlay_cam_on_image,
    preprocess_v2s,
    resolve_sidecar,
)

logger = logging.getLogger(__name__)

# ── XRV constants ────────────────────────────────────────────────────────────
XRV_WEIGHTS = "densenet121-res224-all"
XRV_MODEL_TAG = "xrv-densenet121-all"
XRV_INPUT_SIZE = 224
XRV_MIN = -1024.0
XRV_MAX = 1024.0
XRV_THRESHOLD = 0.5

# Filename TorchXRayVision derives from the release URL. Checked before the
# model is built so a missing file fails fast instead of triggering a download.
XRV_WEIGHTS_FILENAME = (
    "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-"
    "tr15-sc15-seed0-best.pt"
)

# XRV 18-class order, exactly as ``xrv.models.DenseNet(weights=...).pathologies``
# reports it. Verified against the loaded model at load time.
XRV_CLASS_NAMES: tuple[str, ...] = (
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
)


def _encode_png_base64(image_rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _resolve_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    """Pull the parameter dict out of a checkpoint, unwrapping the common envelopes."""
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is a {type(payload).__name__}, expected a dict")
    state = payload.get("state_dict") or payload.get("model") or payload
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a state dict")
    return {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}


class V2SPipeline:
    """Lazily-loaded LungV2S inference wrapper."""

    def __init__(
        self,
        checkpoint: str,
        sidecar: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = checkpoint
        self.sidecar_path = sidecar or resolve_sidecar(checkpoint)
        self._device = device
        self.model: Optional[torch.nn.Module] = None
        self.class_names: tuple[str, ...] = CLASS_NAMES
        self.infer_size: int = DEFAULT_INFER_SIZE
        self.sidecar: dict[str, Any] = {}
        self.checkpoint_epoch: Optional[int] = None
        self.checkpoint_kind: Optional[str] = None
        self.checkpoint_metrics: Optional[dict[str, Any]] = None
        self.load_time_ms: Optional[float] = None
        self.last_error: Optional[str] = None
        self._infer_lock = threading.Lock()

    # ── lifecycle ───────────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = _resolve_device(None)
        elif isinstance(self._device, str):
            self._device = torch.device(self._device)
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.model is not None:
            return
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                f"LungV2S checkpoint not found at {self.checkpoint_path}"
            )

        started = time.perf_counter()
        sidecar = self._read_sidecar()
        class_names = tuple(sidecar["class_names"])
        infer_size = int(sidecar["infer_size"])

        model = build_v2s_model(num_classes=len(class_names), pretrained=False)

        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        declared = payload.get("class_names") if isinstance(payload, dict) else None
        if declared is not None and tuple(declared) != class_names:
            raise RuntimeError(
                f"Checkpoint class order {list(declared)} does not match the sidecar "
                f"order {list(class_names)}; refusing to serve misaligned outputs."
            )

        model.load_state_dict(_extract_state_dict(payload), strict=True)
        model.eval().to(self.device)

        self.model = model
        self.sidecar = sidecar
        self.class_names = class_names
        self.infer_size = infer_size
        if isinstance(payload, dict):
            self.checkpoint_epoch = payload.get("epoch")
            self.checkpoint_kind = payload.get("kind")
            self.checkpoint_metrics = payload.get("metrics")
        self.load_time_ms = (time.perf_counter() - started) * 1000.0
        self.last_error = None
        logger.info(
            "LungV2S loaded in %.0f ms on %s (infer_size=%d, epoch %s, %s)",
            self.load_time_ms,
            self.device,
            self.infer_size,
            self.checkpoint_epoch,
            self.checkpoint_kind,
        )

    def unload(self) -> None:
        self.model = None
        self.load_time_ms = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _read_sidecar(self) -> dict[str, Any]:
        """Load the label-space sidecar, or fail closed.

        The sidecar is the only record of which input size the checkpoint was
        validated at and which class order its head was trained for. Serving a
        384-trained head at 300 px, or against a guessed class order, produces
        confident wrong answers with no error anywhere — so a missing or
        malformed sidecar is a hard failure, not a warning with defaults.
        """
        try:
            return load_sidecar(self.sidecar_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Label-space sidecar not found at {self.sidecar_path}. The Kaggle "
                "run writes lung_v2s.label_space.json beside the checkpoint; copy it "
                "along with the .pth or the model cannot be served safely."
            ) from exc
        except (ValueError, OSError) as exc:
            raise ValueError(
                f"Label-space sidecar at {self.sidecar_path} is unusable: {exc}"
            ) from exc

    # ── inference ───────────────────────────────────────────────────────────

    def predict(self, image: Image.Image) -> dict[str, Any]:
        if self.model is None:
            self.load()
        assert self.model is not None

        with self._infer_lock:
            started = time.perf_counter()
            tensor = preprocess_v2s(image, self.infer_size).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probabilities = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()

            top_index = int(np.argmax(probabilities))
            with GradCAMV2S(self.model) as cam_gen:
                cam = cam_gen.generate(tensor.detach(), top_index)

            rgb = np.asarray(
                image.convert("RGB").resize((self.infer_size, self.infer_size), Image.BICUBIC),
                dtype=np.uint8,
            )
            overlay = overlay_cam_on_image(rgb, cam, alpha=0.42)
            bbox = cam_to_bbox(cam, threshold=0.6)
            if bbox:
                # cam_to_bbox works in CAM-grid pixels (the conv_head feature map,
                # ~10x10 for a 300 px input), not in image pixels. The client draws
                # the box as a fraction of infer_size, so rescale here or every box
                # renders at ~3% of the plate, pinned to the top-left corner.
                cam_h, cam_w = cam.shape[-2], cam.shape[-1]
                sx = self.infer_size / float(cam_w)
                sy = self.infer_size / float(cam_h)
                bbox = (
                    bbox[0] * sx,
                    bbox[1] * sy,
                    bbox[2] * sx,
                    bbox[3] * sy,
                )
            inference_ms = (time.perf_counter() - started) * 1000.0

        return {
            "top_class": self.class_names[top_index],
            "confidence": float(probabilities[top_index]),
            "probabilities": [float(p) for p in probabilities],
            "class_names": list(self.class_names),
            "gradcam_base64": _encode_png_base64(overlay),
            "gradcam_grid": [self.infer_size, self.infer_size],
            "bbox": list(bbox) if bbox else None,
            "inference_ms": inference_ms,
            "model_tag": V2S_MODEL_TAG,
            "infer_size": self.infer_size,
        }

    # ── metadata ────────────────────────────────────────────────────────────

    def checkpoint_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "path": self.checkpoint_path,
            "sidecar": self.sidecar_path,
            "present": os.path.isfile(self.checkpoint_path),
            "sidecar_present": os.path.isfile(self.sidecar_path),
            "loaded": self.is_loaded,
            "epoch": self.checkpoint_epoch,
            "kind": self.checkpoint_kind,
            "metrics": self.checkpoint_metrics,
            "infer_size": self.infer_size,
        }
        try:
            stat = os.stat(self.checkpoint_path)
        except OSError:
            return info
        info["size_mb"] = round(stat.st_size / (1024 * 1024), 1)
        info["modified"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
        return info


class XRVPipeline:
    """Lazily-loaded stock TorchXRayVision DenseNet-121, 18-class multilabel."""

    def __init__(self, cache_dir: str, device: Optional[str] = None) -> None:
        self.cache_dir = cache_dir
        self.weights_path = os.path.join(cache_dir, XRV_WEIGHTS_FILENAME)
        self._device = device
        self.model: Optional[torch.nn.Module] = None
        self.class_names: tuple[str, ...] = XRV_CLASS_NAMES
        self.load_time_ms: Optional[float] = None
        self.last_error: Optional[str] = None
        self._infer_lock = threading.Lock()

    # ── lifecycle ───────────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = _resolve_device(None)
        elif isinstance(self._device, str):
            self._device = torch.device(self._device)
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.model is not None:
            return
        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(
                f"Stock XRV weights not found at {self.weights_path}. They ship with "
                "the repo; serving never downloads them."
            )

        started = time.perf_counter()
        # Imported here, not at module scope: importing this module must not pull
        # torchxrayvision in.
        import torchxrayvision as xrv

        model = xrv.models.DenseNet(
            weights=XRV_WEIGHTS,
            cache_dir=self.cache_dir,
        )
        # The constructor re-installs the published operating points as a buffer
        # whenever the weights entry carries them, which would make forward()
        # apply sigmoid *and* re-normalise against those thresholds — every
        # posterior lands between 0.57 and 0.65 and a 50% cut means nothing.
        # Both are pinned after construction rather than passed as kwargs, so
        # this does not depend on the installed torchxrayvision signature.
        model.op_threshs = None
        model.apply_sigmoid = True

        pathologies = tuple(str(name) for name in model.pathologies)
        if pathologies != XRV_CLASS_NAMES:
            raise RuntimeError(
                f"XRV pathology order {list(pathologies)} does not match the serving "
                f"order {list(XRV_CLASS_NAMES)}; refusing to serve misaligned outputs."
            )

        model.eval().to(self.device)
        self.model = model
        self.class_names = pathologies
        self.load_time_ms = (time.perf_counter() - started) * 1000.0
        self.last_error = None
        logger.info(
            "XRV stock loaded in %.0f ms on %s (%d pathologies)",
            self.load_time_ms,
            self.device,
            len(pathologies),
        )

    def unload(self) -> None:
        self.model = None
        self.load_time_ms = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── inference ───────────────────────────────────────────────────────────

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Grayscale 224x224 scaled to XRV's [-1024, 1024]. Nothing else.

        This is the XRV input contract and it is not shared with LungV2S: no
        channel replication, no ImageNet statistics, no bicubic resize. See the
        module docstring for what happens when the two are crossed.
        """
        gray = image.convert("L").resize((XRV_INPUT_SIZE, XRV_INPUT_SIZE), Image.BILINEAR)
        array = np.asarray(gray, dtype=np.float32) / 255.0
        array = array * (XRV_MAX - XRV_MIN) + XRV_MIN
        tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image: Image.Image, detail: bool = False) -> dict[str, Any]:
        if self.model is None:
            self.load()
        assert self.model is not None

        with self._infer_lock:
            started = time.perf_counter()
            tensor = self._preprocess(image)

            with torch.no_grad():
                probabilities = self.model(tensor)[0].float().cpu().numpy()

            order = np.argsort(-probabilities)
            findings = [
                {
                    "finding": self.class_names[int(index)],
                    "probability": float(probabilities[int(index)]),
                    "rank": rank + 1,
                }
                for rank, index in enumerate(order)
                if float(probabilities[int(index)]) >= XRV_THRESHOLD
            ]

            result: dict[str, Any] = {
                "findings": findings,
                "threshold": XRV_THRESHOLD,
                "count_above_threshold": len(findings),
                "scores": None,
                "gradcam_base64": None,
                "inference_ms": 0.0,
                "model_tag": XRV_MODEL_TAG,
            }

            if detail:
                result["scores"] = [
                    {
                        "finding": self.class_names[int(index)],
                        "probability": float(probabilities[int(index)]),
                    }
                    for index in order
                ]
                top_index = int(order[0])
                with xrv_model.GradCAM(self.model) as cam_gen:
                    cam = cam_gen.generate(tensor.detach(), top_index)
                rgb = np.asarray(
                    image.convert("RGB").resize((XRV_INPUT_SIZE, XRV_INPUT_SIZE), Image.BILINEAR),
                    dtype=np.uint8,
                )
                result["gradcam_base64"] = _encode_png_base64(
                    overlay_cam_on_image(rgb, cam, alpha=0.42)
                )

            result["inference_ms"] = (time.perf_counter() - started) * 1000.0

        return result

    # ── metadata ────────────────────────────────────────────────────────────

    def checkpoint_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "path": self.weights_path,
            "present": os.path.isfile(self.weights_path),
            "loaded": self.is_loaded,
            "weights": XRV_WEIGHTS,
            "pretrained": True,
            "epoch": None,
            "kind": "pretrained",
            "input_size": XRV_INPUT_SIZE,
        }
        try:
            stat = os.stat(self.weights_path)
        except OSError:
            return info
        info["size_mb"] = round(stat.st_size / (1024 * 1024), 1)
        info["modified"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
        return info


class DualLungPipeline:
    """Owns both pipelines. Loads lazily, fails independently, unloads independently."""

    MODELS = ("v2s", "xrv")

    def __init__(
        self,
        v2s_checkpoint: Optional[str] = None,
        v2s_sidecar: Optional[str] = None,
        xrv_cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        base = os.path.dirname(os.path.abspath(__file__))
        checkpoints = os.path.join(base, "checkpoints")
        checkpoint = v2s_checkpoint or os.path.join(checkpoints, CHECKPOINT_NAME)
        cache_dir = xrv_cache_dir or os.path.join(checkpoints, "xrv")
        self.v2s = V2SPipeline(checkpoint, sidecar=v2s_sidecar, device=device)
        self.xrv = XRVPipeline(cache_dir, device=device)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _selected(self, which: str) -> list[tuple[str, Any]]:
        name = (which or "all").strip().lower()
        if name == "all":
            return [("v2s", self.v2s), ("xrv", self.xrv)]
        if name == "v2s":
            return [("v2s", self.v2s)]
        if name == "xrv":
            return [("xrv", self.xrv)]
        raise ValueError(f"Unknown model selector {which!r}; expected 'v2s', 'xrv' or 'all'")

    @staticmethod
    def _describe(exc: BaseException) -> str:
        return f"{type(exc).__name__}: {exc}"

    # ── lifecycle ───────────────────────────────────────────────────────────

    def load(self, which: str = "all") -> dict[str, Optional[str]]:
        """Load the selected models. Never raises for a per-model failure."""
        errors: dict[str, Optional[str]] = {}
        for name, pipeline in self._selected(which):
            try:
                pipeline.load()
                errors[name] = None
            except Exception as exc:  # noqa: BLE001 - reported to the caller
                pipeline.last_error = self._describe(exc)
                errors[name] = pipeline.last_error
                logger.exception("%s load failed", name.upper())
        return errors

    def unload(self, which: str = "all") -> dict[str, bool]:
        unloaded: dict[str, bool] = {}
        for name, pipeline in self._selected(which):
            was_loaded = pipeline.is_loaded
            pipeline.unload()
            pipeline.last_error = None
            unloaded[name] = was_loaded
        return unloaded

    # ── inference ───────────────────────────────────────────────────────────

    def classify(self, image: Image.Image, include_xrv_detail: bool = False) -> dict[str, Any]:
        """Run both models; one failing never takes the other down."""
        started = time.perf_counter()
        v2s_result: Optional[dict[str, Any]] = None
        xrv_result: Optional[dict[str, Any]] = None
        errors: dict[str, Optional[str]] = {"v2s": None, "xrv": None}

        try:
            v2s_result = self.v2s.predict(image)
        except Exception as exc:  # noqa: BLE001
            errors["v2s"] = self._describe(exc)
            self.v2s.last_error = errors["v2s"]
            logger.exception("LungV2S inference failed")

        try:
            xrv_result = self.xrv.predict(image, detail=include_xrv_detail)
        except Exception as exc:  # noqa: BLE001
            errors["xrv"] = self._describe(exc)
            self.xrv.last_error = errors["xrv"]
            logger.exception("XRV inference failed")

        return {
            "v2s": v2s_result,
            "xrv": xrv_result,
            "errors": errors,
            "total_ms": (time.perf_counter() - started) * 1000.0,
        }

    # ── metadata ────────────────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        v2s_info = self.v2s.checkpoint_info()
        xrv_info = self.xrv.checkpoint_info()
        present = {"v2s": bool(v2s_info["present"]), "xrv": bool(xrv_info["present"])}
        if all(present.values()):
            status = "ok"
        elif any(present.values()):
            status = "degraded"
        else:
            status = "unavailable"
        return {
            "status": status,
            "v2s": {
                "loaded": self.v2s.is_loaded,
                "checkpoint_present": present["v2s"],
                "detail": self.v2s.last_error,
            },
            "xrv": {
                "loaded": self.xrv.is_loaded,
                "checkpoint_present": present["xrv"],
                "detail": self.xrv.last_error,
            },
        }

    def model_info(self) -> dict[str, Any]:
        return {
            "v2s": {
                "model_tag": V2S_MODEL_TAG,
                "classes": list(self.v2s.class_names),
                "input_size": self.v2s.infer_size,
                "checkpoint": self.v2s.checkpoint_info(),
            },
            "xrv": {
                "model_tag": XRV_MODEL_TAG,
                "classes": list(XRV_CLASS_NAMES),
                "input_size": XRV_INPUT_SIZE,
                "checkpoint": self.xrv.checkpoint_info(),
            },
        }


__all__ = [
    "V2SPipeline",
    "XRVPipeline",
    "DualLungPipeline",
    "XRV_CLASS_NAMES",
    "XRV_MODEL_TAG",
    "XRV_INPUT_SIZE",
    "XRV_THRESHOLD",
    "CLASS_NAMES",
]
