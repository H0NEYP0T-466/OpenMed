"""Brain classification inference pipeline.

Single-image inference with optional Grad-CAM overlay. The label space bound to
the checkpoint is re-verified at load time, so a head/name disagreement fails
closed instead of silently relabelling predictions.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Optional, Union

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image

from .label_space import (
    CLASS_NAMES,
    LabelSpace,
    LabelSpaceError,
    resolve_label_space,
    split_class_name,
)
from .model import (
    MODEL_TAG,
    GradCAM,
    checkpoint_class_count,
    create_model,
    overlay_cam_on_image,
    resolve_gradcam_layer,
)
from .preprocessor import get_inference_transform, input_side_length

logger = logging.getLogger(__name__)

MAX_IMAGE_SIDE = 8192
MIN_IMAGE_SIDE = 8

__all__ = [
    "CLASS_NAMES",
    "BrainClassificationPipeline",
    "ModelUnavailableError",
]


class ModelUnavailableError(RuntimeError):
    """No usable, label-consistent checkpoint could be prepared for inference."""


class BrainClassificationPipeline:
    """End-to-end inference for the brain tumour classifier."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        allow_untrained_fallback: bool = False,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model_tag = MODEL_TAG

        state_dict = self._load_state_dict(model_path, allow_untrained_fallback)
        detected = checkpoint_class_count(state_dict) if state_dict else None

        try:
            self.label_space: LabelSpace = resolve_label_space(model_path, detected)
        except LabelSpaceError as exc:
            raise ModelUnavailableError(str(exc)) from exc

        self.num_classes = self.label_space.num_classes
        self.class_names: tuple[str, ...] = self.label_space.class_names
        self.declared_model_tag = self.label_space.model_tag or MODEL_TAG
        self.model_tag = self._architecture_tag(self.declared_model_tag)

        try:
            self.model = create_model(
                num_classes=self.num_classes,
                pretrained=state_dict is None,
                model_tag=self.model_tag,
            )
        except (RuntimeError, ValueError) as exc:
            raise ModelUnavailableError(
                f"Could not build backbone {self.model_tag!r} for "
                f"{self.label_space.source}: {exc}"
            ) from exc

        self.using_trained_weights = state_dict is not None

        if state_dict is not None:
            self._populate(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.transform = get_inference_transform(self.model)
        self.input_size = input_side_length(self.model)
        self.target_layer = resolve_gradcam_layer(self.model)

        logger.info(
            "Brain pipeline ready  device=%s  classes=%d  label_source=%s  trained=%s  input=%d",
            self.device,
            self.num_classes,
            self.label_space.source,
            self.using_trained_weights,
            self.input_size,
        )

    @staticmethod
    def _architecture_tag(declared: str) -> str:
        """Only build an architecture the label file names if timm actually knows it.

        A stale or hand-edited sidecar should degrade to the project default with
        a warning, not abort the service with an opaque registry error.
        """
        if declared == MODEL_TAG:
            return declared
        try:
            known = timm.is_model(declared)
        except Exception:  # noqa: BLE001 - registry lookup must never be fatal
            known = False

        if known:
            return declared

        logger.warning(
            "Label file declares unknown architecture %r; falling back to %r.",
            declared,
            MODEL_TAG,
        )
        return MODEL_TAG

    def _load_state_dict(
        self, model_path: Optional[str], allow_untrained_fallback: bool
    ) -> Optional[dict[str, Any]]:
        if not model_path:
            if allow_untrained_fallback:
                logger.warning(
                    "No checkpoint path given; untrained fallback explicitly allowed."
                )
                return None
            raise ModelUnavailableError(
                "No checkpoint path supplied. Refusing to serve random weights."
            )

        if not os.path.isfile(model_path):
            raise ModelUnavailableError(
                f"Brain classifier checkpoint not found at {model_path}. "
                "Train it with brain_kaggle.py and copy the .pth into place."
            )

        try:
            state = torch.load(model_path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - surfaced as a serviceable error
            raise ModelUnavailableError(
                f"Could not read checkpoint {model_path}: {exc}"
            ) from exc

        if not isinstance(state, dict) or not state:
            raise ModelUnavailableError(
                f"Checkpoint {model_path} does not contain a state dict."
            )
        return state

    def _populate(self, state_dict: dict[str, Any]) -> None:
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ModelUnavailableError(
                f"Checkpoint is incompatible with a {self.num_classes}-class "
                f"{self.model_tag} backbone: {exc}"
            ) from exc

        head_width = checkpoint_class_count(self.model.state_dict())
        if head_width is not None and head_width != len(self.class_names):
            raise ModelUnavailableError(
                f"After loading, the head emits {head_width} logits but "
                f"{len(self.class_names)} class names are bound."
            )

    @staticmethod
    def _load_image(image_path_or_pil: Union[str, Image.Image]) -> Image.Image:
        if isinstance(image_path_or_pil, str):
            try:
                image = Image.open(image_path_or_pil)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Unreadable image at {image_path_or_pil}: {exc}") from exc
        elif isinstance(image_path_or_pil, Image.Image):
            image = image_path_or_pil
        else:
            raise TypeError(
                "Expected a filesystem path or PIL.Image, got "
                f"{type(image_path_or_pil).__name__}."
            )

        with image:
            image = image.convert("RGB")
            width, height = image.size
            if min(width, height) < MIN_IMAGE_SIDE:
                raise ValueError(
                    f"Image is too small to classify ({width}x{height})."
                )
            if max(width, height) > MAX_IMAGE_SIDE:
                raise ValueError(
                    f"Image exceeds the {MAX_IMAGE_SIDE}px side limit ({width}x{height})."
                )
            return image.copy()

    def _forward_logits(self, image: Image.Image) -> torch.Tensor:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(tensor)

    def _interpret(self, logits: torch.Tensor) -> dict[str, Any]:
        probs = F.softmax(logits, dim=1)[0].detach().cpu()
        top_prob, top_idx = torch.max(probs, dim=0)
        k = min(5, self.num_classes)
        top5_probs, top5_idxs = torch.topk(probs, k)

        predicted_class = self.class_names[int(top_idx.item())]
        tumor_type, sequence = split_class_name(predicted_class)

        return {
            "predicted_class": predicted_class,
            "confidence": round(float(top_prob.item()), 4),
            "tumor_type": tumor_type,
            "sequence": sequence,
            "top5": [
                {
                    "class": self.class_names[int(idx.item())],
                    "confidence": round(float(prob.item()), 4),
                }
                for idx, prob in zip(top5_idxs, top5_probs, strict=True)
            ],
            "probabilities": [round(float(value), 6) for value in probs.tolist()],
            "model_trained": self.using_trained_weights,
            "label_space_source": self.label_space.source,
        }

    def predict(self, image_path_or_pil: Union[str, Image.Image]) -> dict[str, Any]:
        image = self._load_image(image_path_or_pil)
        return self._interpret(self._forward_logits(image))

    def predict_with_gradcam(
        self, image_path_or_pil: Union[str, Image.Image]
    ) -> dict[str, Any]:
        """Inference plus a Grad-CAM overlay rendered at model input resolution."""
        image = self._load_image(image_path_or_pil)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with GradCAM(self.model, self.target_layer) as grad_cam:
            cam = grad_cam.generate(tensor)
            logits = grad_cam.logits
            if logits is None:  # pragma: no cover - generate always sets it
                raise RuntimeError("Grad-CAM produced no logits to interpret.")

        result = self._interpret(logits)
        base = np.array(image.resize((self.input_size, self.input_size)))
        overlay = overlay_cam_on_image(base, cam)
        result["gradcam_base64"] = _encode_jpeg_data_url(overlay)
        result["gradcam_grid"] = [int(cam.shape[0]), int(cam.shape[1])]
        result["explainability"] = {
            "method": "gradcam",
            "layer": type(self.target_layer).__name__,
            "interpretation": (
                "Coarse activation map at model input resolution. Indicates "
                "regions the classifier relied on; not a lesion segmentation."
            ),
        }
        return result

    def model_info(self) -> dict[str, Any]:
        return {
            "model_name": "EfficientNetV2-B2",
            "model_tag": self.model_tag,
            "declared_model_tag": self.declared_model_tag,
            "architecture_matches_declaration": self.model_tag == self.declared_model_tag,
            "num_classes": self.num_classes,
            "class_names": list(self.class_names),
            "input_size": f"{self.input_size}x{self.input_size}",
            "label_space_source": self.label_space.source,
            "trained_weights_loaded": self.using_trained_weights,
            "metrics": self.label_space.metrics,
        }


def _encode_jpeg_data_url(array_rgb: np.ndarray, quality: int = 90) -> str:
    encoded, buffer = cv2.imencode(
        ".jpg",
        cv2.cvtColor(array_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not encoded:
        raise RuntimeError("JPEG encoding failed for the Grad-CAM overlay.")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")
