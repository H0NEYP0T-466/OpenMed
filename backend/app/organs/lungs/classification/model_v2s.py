"""LungV2S — EfficientNetV2-S four-class lung classifier, and *its* preprocessing.

This module owns exactly one preprocessing contract:

    grayscale -> replicated to 3 channels -> resize to ``infer_size`` (300 or 384)
    -> ImageNet mean/std normalisation -> float32 NCHW in roughly [-2.1, 2.6]

The stock XRV DenseNet in the sibling pipeline uses a completely different one:

    grayscale -> single channel -> resize to 224 -> scale to [-1024, 1024]

The two are not interchangeable and no code is shared between them on purpose.
Feeding an XRV tensor to this model, or a LungV2S tensor to XRV, does not raise,
warn, or produce obviously broken output. Both networks are pure functions of
whatever tensor they receive; they will happily run and emit a confident
softmax/sigmoid over the wrong input distribution. The failure is silent: a
[-1024, 1024] grayscale tensor pushed through ImageNet normalisation lands
almost entirely in saturated negative territory, and an ImageNet-normalised
RGB tensor fed to XRV is interpreted as a near-uniform grey image. Either way
the reported probabilities look plausible and are meaningless. That is why the
normalisation constants and the tensor construction live here, beside the
architecture, and are never parameterised from outside.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_TAG = "tf_efficientnetv2_s.in21k_ft_in1k"

# Fixed order. The classifier head's output index maps straight onto this tuple;
# the training package writes the same list into the sidecar and the checkpoint.
CLASS_NAMES: tuple[str, ...] = ("COVID", "Lung_Opacity", "Normal", "Viral Pneumonia")
NUM_CLASSES = len(CLASS_NAMES)

# Training resolution, and the two inference resolutions the TTA sweep chooses
# between. Anything else is a mistake, not a knob.
TRAIN_SIZE = 300
DEFAULT_INFER_SIZE = 300
SUPPORTED_INFER_SIZES: tuple[int, ...] = (300, 384)

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

CHECKPOINT_NAME = "lung_v2s_best.pth"
SIDECAR_NAME = "lung_v2s.label_space.json"

_MEAN_ARRAY = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
_STD_ARRAY = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)


def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    drop_path_rate: float = 0.0,
    drop_rate: Optional[float] = None,
) -> nn.Module:
    """Build the timm EfficientNetV2-S backbone with a head of ``num_classes``.

    ``pretrained`` downloads ImageNet weights and is therefore only ever true on
    a training box. Serving passes ``False`` because the checkpoint replaces
    every parameter anyway, and the serving box has no network access to the
    timm weight host.
    """
    import timm

    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    kwargs: dict[str, Any] = {"drop_path_rate": drop_path_rate}
    if drop_rate is not None:
        kwargs["drop_rate"] = drop_rate
    return timm.create_model(
        MODEL_TAG, pretrained=pretrained, num_classes=num_classes, **kwargs
    )


def resolve_sidecar(checkpoint_path: str) -> str:
    """Locate ``lung_v2s.label_space.json`` next to the checkpoint or its parent.

    Training writes the sidecar at ``<output_dir>/lung_v2s.label_space.json``
    while the checkpoint lands in ``<output_dir>/checkpoints/``; serving keeps
    both side by side. Both layouts are accepted.
    """
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    beside = os.path.join(checkpoint_dir, SIDECAR_NAME)
    if os.path.isfile(beside):
        return beside
    parent = os.path.join(os.path.dirname(checkpoint_dir), SIDECAR_NAME)
    if os.path.isfile(parent):
        return parent
    return beside


def load_sidecar(path: Optional[str] = None) -> dict[str, Any]:
    """Read and validate the label-space sidecar.

    Validation is strict on the two fields that can silently corrupt output —
    class order and inference resolution — and permissive on the rest, which is
    reporting metadata that may legitimately be absent on a fresh run.
    """
    if not path:
        raise ValueError("load_sidecar requires a path")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Sidecar at {path} is not a JSON object")

    declared = payload.get("class_names")
    if declared is not None:
        names = tuple(str(name) for name in declared)
        if names != CLASS_NAMES:
            raise ValueError(
                f"Sidecar class order {list(names)} does not match the serving order "
                f"{list(CLASS_NAMES)}; refusing to serve misaligned outputs."
            )

    infer_size = payload.get("infer_size", payload.get("input_size", DEFAULT_INFER_SIZE))
    try:
        infer_size = int(infer_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Sidecar infer_size {infer_size!r} is not an integer") from exc
    if infer_size not in SUPPORTED_INFER_SIZES:
        raise ValueError(
            f"Sidecar infer_size {infer_size} is not one of {SUPPORTED_INFER_SIZES}"
        )

    num_classes = payload.get("num_classes", NUM_CLASSES)
    if int(num_classes) != NUM_CLASSES:
        raise ValueError(
            f"Sidecar declares {num_classes} classes; this build serves {NUM_CLASSES}"
        )

    resolved = dict(payload)
    resolved["class_names"] = list(CLASS_NAMES)
    resolved["num_classes"] = NUM_CLASSES
    resolved["infer_size"] = infer_size
    resolved["input_size"] = int(payload.get("input_size", TRAIN_SIZE))
    return resolved


def preprocess_v2s(image: Image.Image, infer_size: int = DEFAULT_INFER_SIZE) -> torch.Tensor:
    """Turn a PIL image into the exact tensor LungV2S was trained on.

    Returns a CPU float32 tensor of shape ``(1, 3, infer_size, infer_size)``.
    Kept free of torchvision transforms so the serving path cannot drift from
    the training transform through a library default change.
    """
    if infer_size not in SUPPORTED_INFER_SIZES:
        raise ValueError(
            f"infer_size must be one of {SUPPORTED_INFER_SIZES}, got {infer_size}"
        )
    if not isinstance(image, Image.Image):
        raise TypeError(f"preprocess_v2s expects a PIL image, got {type(image)!r}")

    grayscale = image.convert("L")
    resized = grayscale.resize((infer_size, infer_size), Image.BICUBIC)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    rgb = np.repeat(pixels[:, :, None], 3, axis=2)
    normalised = (rgb - _MEAN_ARRAY) / _STD_ARRAY
    chw = np.ascontiguousarray(normalised.transpose(2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def resolve_cam_layer(model: nn.Module) -> nn.Module:
    """Return the timm ``conv_head`` module used as the Grad-CAM target."""
    layer = getattr(model, "conv_head", None)
    if not isinstance(layer, nn.Module):
        raise RuntimeError(
            f"Model {type(model).__name__} has no 'conv_head' module; Grad-CAM for "
            "LungV2S is defined on the final EfficientNet convolution."
        )
    return layer


class GradCAMV2S:
    """Grad-CAM for the timm EfficientNetV2-S backbone.

    The activation is captured by a *forward* hook that calls ``retain_grad`` on
    the convolution output. The output must be kept as-is rather than cloned:
    EfficientNetV2 routes it through ``bn2`` and an in-place SiLU, so a full
    backward hook on the convolution output would be handed a tensor autograd
    has already marked as modified in place, and a clone would be a sibling
    branch of the loss rather than an ancestor of it, leaving
    ``autograd.grad`` with an unused tensor. ``retain_grad`` keeps the real
    graph edge intact, so the gradient of the target logit with respect to the
    convolution map is well defined.

    Hooks stay attached for the object's lifetime; use the instance as a context
    manager to guarantee they are removed.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.target_layer = resolve_cam_layer(model)
        self.activations: Optional[torch.Tensor] = None
        self._handles: list[Any] = [self.target_layer.register_forward_hook(self._save_activation)]

    def _save_activation(self, module: nn.Module, inputs: Any, output: Any) -> None:
        if isinstance(output, (tuple, list)):
            output = output[0]
        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.retain_grad()
        self.activations = output

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "GradCAMV2S":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.remove_hooks()

    def generate(self, tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Return a normalised ``(H, W)`` activation map for one image."""
        if tensor.dim() != 4 or tensor.shape[0] != 1:
            raise ValueError(
                f"GradCAMV2S expects a single-image batch, got shape {tuple(tensor.shape)}"
            )

        was_training = self.model.training
        self.model.eval()
        self.activations = None
        try:
            self.model.zero_grad(set_to_none=True)
            if not tensor.requires_grad:
                tensor = tensor.clone().detach().requires_grad_(True)

            logits = self.model(tensor)
            if self.activations is None:
                raise RuntimeError("Grad-CAM forward hook captured no activation")
            if not self.activations.requires_grad:
                raise RuntimeError(
                    "Captured activation carries no gradient graph; Grad-CAM must run "
                    "outside torch.no_grad()."
                )

            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())

            gradients = torch.autograd.grad(logits[0, int(target_class)], self.activations)[0]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * self.activations).sum(dim=1)).squeeze(0)
            array = cam.detach().float().cpu().numpy()

            array = array - float(array.min())
            peak = float(array.max())
            if peak > 0:
                array = array / peak
            return array
        finally:
            self.model.zero_grad(set_to_none=True)
            if was_training:
                self.model.train()


try:  # pragma: no cover - the sibling module always exists in the app tree
    from .model import cam_to_bbox, overlay_cam_on_image
except ImportError:  # standalone import (tests, one-off scripts)
    import cv2

    def overlay_cam_on_image(
        image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4
    ) -> np.ndarray:
        array = np.asarray(image_rgb)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 RGB image, got shape {array.shape}")
        cam = np.asarray(cam, dtype=np.float32)
        if cam.ndim != 2 or cam.size == 0:
            return array
        resized = cv2.resize(
            cam, (array.shape[1], array.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        peak = float(resized.max())
        if peak <= 0:
            return array
        heatmap = cv2.applyColorMap(np.uint8(255.0 * resized / peak), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        base = array.astype(np.float32) / 255.0
        blended = np.clip(alpha * heatmap + (1.0 - alpha) * base, 0.0, 1.0)
        return (blended * 255.0).astype(np.uint8)

    def cam_to_bbox(
        cam: np.ndarray, threshold: float = 0.6
    ) -> Optional[tuple[int, int, int, int]]:
        if cam.size == 0:
            return None
        mask = cam >= threshold
        if not mask.any():
            return None
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


__all__ = [
    "MODEL_TAG",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "TRAIN_SIZE",
    "DEFAULT_INFER_SIZE",
    "SUPPORTED_INFER_SIZES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "CHECKPOINT_NAME",
    "SIDECAR_NAME",
    "build_model",
    "load_sidecar",
    "resolve_sidecar",
    "preprocess_v2s",
    "resolve_cam_layer",
    "GradCAMV2S",
    "overlay_cam_on_image",
    "cam_to_bbox",
]
