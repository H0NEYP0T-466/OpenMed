"""Grad-CAM for the TorchXRayVision DenseNet-121.

TorchXRayVision's `features2` applies `F.relu(features, inplace=True)` to the
output of `features.norm5`. A `register_full_backward_hook` on norm5 makes that
output a view owned by the hook, and the in-place ReLU then raises:

    RuntimeError: Output 0 of BackwardHookFunction is a view and is being
    modified inplace.

So this module never installs a backward hook. It captures the activation with
a forward hook and asks autograd for d(score)/d(activation) directly, which is
the same quantity Grad-CAM needs and cannot collide with in-place ops.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
import torch


class GradCAM:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        target = model.features.norm5
        self._handle = target.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output

    def remove_hooks(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc_info) -> None:
        self.remove_hooks()

    def generate(self, tensor: torch.Tensor, class_index: int) -> np.ndarray:
        if tensor.dim() != 4 or tensor.shape[0] != 1:
            raise ValueError("GradCAM expects a single-image batch")
        if not tensor.requires_grad:
            tensor.requires_grad_(True)

        was_training = self.model.training
        self.model.eval()
        self.activations = None
        try:
            logits = self.model(tensor)
            if self.activations is None:
                raise RuntimeError("Grad-CAM forward hook captured no activation")

            score = logits[0, int(class_index)]
            gradients = torch.autograd.grad(
                outputs=score,
                inputs=self.activations,
                retain_graph=False,
                create_graph=False,
            )[0]

            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * self.activations).sum(dim=1))
            cam = cam.squeeze(0).detach().float().cpu().numpy()
            cam -= float(cam.min())
            peak = float(cam.max())
            if peak > 0:
                cam /= peak
            return cam
        finally:
            if was_training:
                self.model.train()
            self.model.zero_grad(set_to_none=True)


def overlay(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    array = np.asarray(image_rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {array.shape}")
    resized = cv2.resize(np.asarray(cam, dtype=np.float32),
                         (array.shape[1], array.shape[0]), interpolation=cv2.INTER_LINEAR)
    peak = float(resized.max())
    if peak <= 0:
        return array
    heat = cv2.applyColorMap(np.uint8(255.0 * resized / peak), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    base = array.astype(np.float32) / 255.0
    return (np.clip(alpha * heat + (1.0 - alpha) * base, 0.0, 1.0) * 255.0).astype(np.uint8)
