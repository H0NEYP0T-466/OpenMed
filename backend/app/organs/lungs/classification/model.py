"""Z-Net for 14-label chest radiograph classification (Ahmad et al., Diagnostics 2023).

Architecture, verbatim from the paper (Section 3, Figure 1):

    DenseNet-121 backbone (ImageNet weights)
      -> delete the fully-connected / classification layers
      -> transition layer   : 1x1 conv, S x S x D activations (D = 224)
      -> prediction layer   : 1x1 conv, S x S x C class logits
      -> global pooling     : LogSumExp over the spatial grid
      -> loss               : weighted binary cross-entropy

The transition x prediction product is the paper's class activation map
(Eq. 3), used there for bounding-box localisation. This module exposes that
map directly (`forward(..., return_maps=True)`) and also provides a classical
Grad-CAM over the last dense block, which is what the serving UI renders.

Keep `ZNet` numerically identical to the copy inside `lung_kaggle.py`; the
Kaggle script is self-contained so it can be uploaded as a single file.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_TAG = "znet_densenet121_224"

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
)

NIH_FINDINGS = FINDINGS  # historical alias
FINDING_INDEX = {name: i for i, name in enumerate(FINDINGS)}


# Paper Table 4 / text, for the per-class comparison chart in training artefacts.
PAPER_AUC: dict[str, float] = {
    "Atelectasis": 0.8200,
    "Cardiomegaly": 0.8725,
    "Effusion": 0.8800,
    "Infiltration": 0.7220,
    "Mass": 0.8406,
    "Pneumonia": 0.7500,
    "Pneumothorax": 0.8980,
    "Hernia": 0.7600,
}
PAPER_MEAN_AUC = 0.858


class ZNet(nn.Module):
    """DenseNet-121 with the paper's transition/prediction head and LSE pooling."""

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        transition_dim: int = 224,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        from torchvision.models import DenseNet121_Weights, densenet121

        backbone = densenet121(
            weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = backbone.features  # (B, 1024, H/32, W/32)
        self.transition = nn.Conv2d(1024, transition_dim, kernel_size=1)
        self.prediction = nn.Conv2d(transition_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.num_classes = num_classes

        for module in (self.transition, self.prediction):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, return_maps: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        fmap = F.relu(self.features(x))
        transition = F.relu(self.transition(fmap))          # B, D, S, S
        class_maps = self.prediction(transition)            # B, C, S, S
        logits = torch.logsumexp(
            class_maps.flatten(2), dim=2
        )                                                   # B, C  (LSE pooling)
        logits = self.dropout(logits)
        if return_maps:
            return logits, class_maps
        return logits

    @torch.no_grad()
    def class_activation(self, x: torch.Tensor, class_index: int) -> np.ndarray:
        """Paper CAM: prediction-layer response for one class, normalised to [0, 1]."""
        _, class_maps = self.forward(x, return_maps=True)
        cam = class_maps[0, int(class_index)].float().cpu().numpy()
        cam = cam - float(cam.min())
        peak = float(cam.max())
        if peak > 0:
            cam /= peak
        return cam


class GradCAM:
    """Gradient-weighted CAM over the final dense block (serving visualisation)."""

    def __init__(self, model: ZNet) -> None:
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles: list = []
        target = model.features.norm5
        self._handles = [
            target.register_forward_hook(self._save_activation),
            target.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

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
        if not tensor.requires_grad:
            tensor.requires_grad_(True)
        was_training = self.model.training
        self.model.eval()
        self.activations = None
        self.gradients = None
        try:
            logits = self.model(tensor)
            self.model.zero_grad(set_to_none=True)
            logits[0, int(class_index)].backward()
            if self.activations is None or self.gradients is None:
                raise RuntimeError("Grad-CAM hooks captured nothing")
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
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
