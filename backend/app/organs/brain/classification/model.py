"""Model definition, optimisation helpers and Grad-CAM explanation utility."""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_TAG = "tf_efficientnetv2_b2.in1k"
GRADCAM_LAYER_CANDIDATES = ("conv_head", "bn2", "stem")


def create_model(
    num_classes: int,
    pretrained: bool = True,
    model_tag: str = MODEL_TAG,
    drop_path_rate: float = 0.0,
    drop_rate: Optional[float] = None,
) -> nn.Module:
    """Build a timm classification backbone with a head sized for `num_classes`.

    `drop_path_rate` enables stochastic depth, the primary regulariser for
    EfficientNet backbones; it is identity in eval mode, so serving is
    unaffected. `drop_rate` overrides the classifier-head dropout probability
    when the architecture default is not wanted.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    kwargs: dict[str, Any] = {"drop_path_rate": drop_path_rate}
    if drop_rate is not None:
        kwargs["drop_rate"] = drop_rate
    return timm.create_model(
        model_tag, pretrained=pretrained, num_classes=num_classes, **kwargs
    )


def head_weight_keys(model: nn.Module) -> tuple[str, ...]:
    """Names of the classifier weight tensors used to size-check a checkpoint."""
    return tuple(
        name
        for name, _ in model.named_parameters()
        if name.endswith(("classifier.weight", "head.fc.weight"))
    )


def checkpoint_class_count(state_dict: dict[str, Any]) -> Optional[int]:
    """Infer the class count a state dict was trained for, or None if undetectable."""
    for suffix in ("classifier.weight", "head.fc.weight"):
        for name, tensor in state_dict.items():
            if name.endswith(suffix) and hasattr(tensor, "shape") and len(tensor.shape) == 2:
                return int(tensor.shape[0])
    return None


def resolve_gradcam_layer(model: nn.Module) -> nn.Module:
    """Return the final convolutional module suitable for Grad-CAM attribution."""
    for name in GRADCAM_LAYER_CANDIDATES:
        layer = getattr(model, name, None)
        if isinstance(layer, nn.Module):
            return layer

    convs = [
        module
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, timm.layers.ConvNormAct2d))
    ]
    if not convs:
        raise RuntimeError("No convolution layer available for Grad-CAM attribution.")
    return convs[-1]


def get_loss_function(
    class_weights: Optional[torch.Tensor] = None,
    device: Any = "cpu",
) -> nn.CrossEntropyLoss:
    if class_weights is None:
        return nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss(weight=class_weights.to(device))


def weighted_soft_target_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy against mixture targets (MixUp/CutMix), still class-weighted.

    Each sample's loss is the target-distribution expectation of the per-class
    weighted NLL, so the rare-class up-weighting survives label mixing instead
    of being washed out by soft targets. The weighted mean matches
    `nn.CrossEntropyLoss(weight=..., reduction="mean")`, which normalises by
    the sum of sample weights rather than the batch size, so train and val
    losses stay on the same scale.
    """
    log_probs = F.log_softmax(logits, dim=1)
    per_sample = -(soft_targets * log_probs).sum(dim=1)
    weights = soft_targets @ class_weights.to(logits.device)
    total_weight = weights.sum()
    if float(total_weight) > 0:
        return (per_sample * weights).sum() / total_weight
    return per_sample.mean()


def get_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.AdamW:
    """AdamW with the conventional exemption of norm and bias terms from decay."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    steps_per_epoch: int,
    max_lr: Optional[float] = None,
    pct_start: float = 0.3,
) -> torch.optim.lr_scheduler.OneCycleLR:
    """One-cycle schedule stepped once per minibatch.

    `pct_start` controls the warmup share of total steps, so warmup is expressed
    in the unit the scheduler actually reasons in rather than in whole epochs.
    """
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be at least 1")
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr or optimizer.param_groups[0]["lr"],
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
    )


class GradCAM:
    """Gradient-weighted Class Activation Mapping for one convolutional layer.

    Hooks stay attached for the object's lifetime so repeated `generate` calls on
    the same instance each read fresh activations; use it as a context manager to
    guarantee removal.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.logits: Optional[torch.Tensor] = None
        self._handles: list[Any] = []
        self._attach()

    def _save_activation(self, module: nn.Module, inputs: Any, output: Any) -> None:
        if isinstance(output, (tuple, list)):
            output = output[0]
        self.activations = output

    def _save_gradient(self, module: nn.Module, grad_input: Any, grad_output: Any) -> None:
        self.gradients = grad_output[0]

    def _attach(self) -> None:
        if self._handles:
            return
        self._handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> GradCAM:
        self._attach()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.remove_hooks()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Return a normalised 2D activation map for a single-image batch."""
        if input_tensor.dim() != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                f"GradCAM expects a batch of exactly one image, got shape "
                f"{tuple(input_tensor.shape)}"
            )

        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True)

        self._attach()
        was_training = self.model.training
        self.model.eval()
        self.activations = None
        self.gradients = None

        try:
            logits = self.model(input_tensor)
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())

            self.model.zero_grad(set_to_none=True)
            logits[0, int(target_class)].backward()
            self.logits = logits.detach()

            if self.activations is None or self.gradients is None:
                raise RuntimeError(
                    "Grad-CAM capture hooks fired no tensors; the target layer "
                    "may not participate in the forward pass."
                )

            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * self.activations).sum(dim=1))
            cam = cam.squeeze(0).detach().to("cpu", dtype=torch.float32).numpy()

            cam = cam - float(cam.min())
            peak = float(cam.max())
            if peak > 0:
                cam = cam / peak
            return cam
        finally:
            if was_training:
                self.model.train()
            self.model.zero_grad(set_to_none=True)


def overlay_cam_on_image(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a normalised activation map onto an RGB uint8 image.

    Shared by the serving pipeline and the training artefacts so an explanation
    is rendered identically in both.
    """
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

    heatmap = cv2.applyColorMap(
        np.uint8(255.0 * resized / peak), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    base = array.astype(np.float32) / 255.0
    blended = np.clip(alpha * heatmap + (1.0 - alpha) * base, 0.0, 1.0)
    return (blended * 255.0).astype(np.uint8)
