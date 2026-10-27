"""XRV DenseNet-121 fine-tune — 16-class chest radiograph classifier.

The checkpoint is a TorchXRayVision ``densenet121-res224-all`` backbone whose
classification head was replaced by ``Dropout(0.3) + Linear(1024, 16)`` and
fine-tuned on the combined NIH + COVID + TB corpus.

Input convention is XRV's, not ImageNet's: one grayscale channel scaled to
[-1024, 1024]. Feeding RGB or ImageNet-normalised tensors produces confident
nonsense — the features are meaningless outside that range.

Grad-CAM hooks ``features.norm5`` with a *forward* hook and asks autograd for
the gradient. A backward hook there crashes: ``features2`` applies an in-place
ReLU to that very output, and PyTorch forbids modifying a hook-owned view.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Order must match the checkpoint's class list exactly — the head's output
# index maps straight onto this tuple.
FINDINGS: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
    "COVID",
    "TB",
)
FINDING_INDEX = {name: index for index, name in enumerate(FINDINGS)}
N_FINDINGS = len(FINDINGS)

XRV_WEIGHTS = "densenet121-res224-all"
INPUT_SIZE = 224
XRV_MIN = -1024.0
XRV_MAX = 1024.0
MODEL_TAG = "xrv-densenet121-16"

# The 14 NIH findings are reported separately from the two added classes: the
# group mean is the number comparable with published ChestX-ray14 baselines.
NIH_FINDINGS = FINDINGS[:14]


def build_model(cache_dir: Optional[str] = None) -> nn.Module:
    """DenseNet-121 with the 16-class head, ready for ``load_state_dict``.

    The TorchXRayVision module is returned unwrapped so the checkpoint's key
    names (``features.*``, ``classifier.1.*``) line up without remapping.
    """
    import torchxrayvision as xrv

    model = xrv.models.DenseNet(weights=XRV_WEIGHTS, cache_dir=cache_dir)
    in_features = int(model.classifier.in_features)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, N_FINDINGS))
    # forward() must return raw logits: op_threshs would apply sigmoid plus
    # operating-point normalisation, which drags every posterior toward 0.5.
    model.op_threshs = None
    model.apply_sigmoid = False
    return model


class GradCAM:
    """Grad-CAM on XRV's final normalisation layer."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        target = getattr(model.features, "norm5", model.features)
        self._handles = [target.register_forward_hook(self._save_activation)]

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc_info) -> None:
        self.remove_hooks()

    def generate(self, tensor: torch.Tensor, class_index: int) -> np.ndarray:
        if tensor.dim() != 4 or tensor.shape[0] != 1:
            raise ValueError("GradCAM expects a single-image batch")
        was_training = self.model.training
        self.model.eval()
        self.activations = None
        try:
            self.model.zero_grad(set_to_none=True)
            if not tensor.requires_grad:
                tensor = tensor.clone().requires_grad_(True)
            logits = self.model(tensor)
            if self.activations is None:
                raise RuntimeError("Grad-CAM forward hook captured no activation")
            gradients = torch.autograd.grad(logits[0, int(class_index)], self.activations)[0]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * F.relu(self.activations)).sum(dim=1)).squeeze(0)
            cam = cam.detach().float().cpu().numpy()
            cam -= float(cam.min())
            peak = float(cam.max())
            if peak > 0:
                cam /= peak
            return cam
        finally:
            self.model.zero_grad(set_to_none=True)
            if was_training:
                self.model.train()


def overlay_cam_on_image(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a normalised activation map onto an RGB uint8 image."""
    array = np.asarray(image_rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {array.shape}")
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2 or cam.size == 0:
        return array
    resized = cv2.resize(cam, (array.shape[1], array.shape[0]), interpolation=cv2.INTER_LINEAR)
    peak = float(resized.max())
    if peak <= 0:
        return array
    heatmap = cv2.applyColorMap(np.uint8(255.0 * resized / peak), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    base = array.astype(np.float32) / 255.0
    return (np.clip(alpha * heatmap + (1.0 - alpha) * base, 0.0, 1.0) * 255.0).astype(np.uint8)


def cam_to_bbox(cam: np.ndarray, threshold: float = 0.6) -> tuple[int, int, int, int] | None:
    """Bounding box of the supra-threshold heat region, in CAM pixel units."""
    if cam.size == 0:
        return None
    mask = cam >= threshold
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


__all__ = [
    "FINDINGS",
    "FINDING_INDEX",
    "NIH_FINDINGS",
    "N_FINDINGS",
    "MODEL_TAG",
    "INPUT_SIZE",
    "XRV_MIN",
    "XRV_MAX",
    "XRV_WEIGHTS",
    "build_model",
    "GradCAM",
    "overlay_cam_on_image",
    "cam_to_bbox",
]
