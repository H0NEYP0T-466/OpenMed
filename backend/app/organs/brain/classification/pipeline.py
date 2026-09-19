"""
Brain Classification Inference Pipeline.

Handles single-image inference with optional Grad-CAM overlay.
Loads the trained EfficientNetV2-B2 checkpoint and applies the
correct timm preprocessing transforms at inference time.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import base64
import io
import logging
from typing import Union, Optional

from .model import create_model, GradCAM
from .preprocessor import get_inference_transform

logger = logging.getLogger(__name__)

# ── Actual 42 classes from the dataset, sorted alphabetically ────────────
# Must match the order produced by BrainTumorDataset.CLASS_NAMES
CLASS_NAMES: list[str] = sorted([
    "Astrocytoma T1", "Astrocytoma T1C+", "Astrocytoma T2",
    "Dysembryoplastic Neuroepithelial Tumor T1",
    "Dysembryoplastic Neuroepithelial Tumor T1C+",
    "Dysembryoplastic Neuroepithelial Tumor T2",
    "Ependymoma - Subependymoma T1",
    "Ependymoma - Subependymoma T1C+",
    "Ependymoma - Subependymoma T2",
    "Ganglioglioma T1", "Ganglioglioma T1C+", "Ganglioglioma T2",
    "Germinoma T1", "Germinoma T1C+", "Germinoma T2",
    "Glioblastoma T1", "Glioblastoma T1C+", "Glioblastoma T2",
    "Hemangiopericytoma T1", "Hemangiopericytoma T1C+", "Hemangiopericytoma T2",
    "Medulloblastoma T1", "Medulloblastoma T1C+", "Medulloblastoma T2",
    "Meningioma T1", "Meningioma T1C+", "Meningioma T2",
    "Neurocytoma T1", "Neurocytoma T1C+", "Neurocytoma T2",
    "Normal T1", "Normal T1C+", "Normal T2",
    "Oligodendroglioma T1", "Oligodendroglioma T1C+", "Oligodendroglioma T2",
    "Pituitary T1", "Pituitary T1C+", "Pituitary T2",
    "Schwannoma T1", "Schwannoma T1C+", "Schwannoma T2",
])

# Map from full class name → (tumor_type, sequence)
_TUMOR_SEQUENCE_MAP: dict[str, tuple[str, str]] = {}
for _cls in CLASS_NAMES:
    # Sequences are always the last token: T1, T1C+, or T2
    _parts = _cls.rsplit(" ", 1)
    if len(_parts) == 2:
        _TUMOR_SEQUENCE_MAP[_cls] = (_parts[0], _parts[1])
    else:
        _TUMOR_SEQUENCE_MAP[_cls] = (_cls, "Unknown")


class BrainClassificationPipeline:
    """
    End-to-end inference pipeline for the brain tumor classifier.

    Usage::

        pipe = BrainClassificationPipeline("checkpoints/brain_best_model.pth")
        result = pipe.predict("scan.jpg")          # basic prediction
        result = pipe.predict_with_gradcam("scan.jpg")  # + Grad-CAM overlay
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = len(CLASS_NAMES)
        self.class_names = CLASS_NAMES

        logger.info(
            f"Initializing BrainClassificationPipeline  "
            f"device={self.device}  classes={self.num_classes}"
        )

        # Build model (pretrained=False — we'll load our own weights)
        self.model = create_model(
            num_classes=self.num_classes, pretrained=(model_path is None)
        )
        self.model.to(self.device)

        if model_path:
            try:
                state_dict = torch.load(
                    model_path, map_location=self.device, weights_only=True
                )
                # Detect checkpoint class count if different
                for k in ["classifier.weight", "head.fc.weight"]:
                    if k in state_dict:
                        ckpt_classes = state_dict[k].shape[0]
                        if ckpt_classes != self.num_classes:
                            self.num_classes = ckpt_classes
                            self.model = create_model(num_classes=self.num_classes, pretrained=False)
                            self.model.to(self.device)
                        break
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded checkpoint  path={model_path}  classes={self.num_classes}")
            except Exception as exc:
                logger.error(f"Failed to load checkpoint: {exc}")
        else:
            logger.warning(
                "No model_path provided — running with ImageNet-pretrained weights "
                "(predictions will be nonsensical until a trained checkpoint is loaded)."
            )

        self.model.eval()
        self.transform = get_inference_transform(self.model)

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _load_image(image_path_or_pil: Union[str, Image.Image]) -> Image.Image:
        if isinstance(image_path_or_pil, str):
            return Image.open(image_path_or_pil).convert("RGB")
        return image_path_or_pil.convert("RGB")

    def _parse_class(self, class_name: str) -> tuple[str, str]:
        """Return (tumor_type, sequence) for a class name."""
        return _TUMOR_SEQUENCE_MAP.get(class_name, (class_name, "Unknown"))

    # ── public API ────────────────────────────────────────────────────────

    def predict(self, image_path_or_pil: Union[str, Image.Image]) -> dict:
        """
        Run inference on a single image.

        Returns
        -------
        dict with keys:
            predicted_class, confidence, tumor_type, sequence,
            top5  (list[{class, confidence}]),
            probabilities  (list[float] of length num_classes)
        """
        image = self._load_image(image_path_or_pil)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        top_prob, top_idx = torch.max(probs, dim=0)
        top5_probs, top5_idxs = torch.topk(probs, min(5, self.num_classes))

        pred_class = self.class_names[top_idx.item()]
        tumor_type, sequence = self._parse_class(pred_class)

        return {
            "predicted_class": pred_class,
            "confidence": round(top_prob.item(), 4),
            "tumor_type": tumor_type,
            "sequence": sequence,
            "top5": [
                {
                    "class": self.class_names[c.item()],
                    "confidence": round(p.item(), 4),
                }
                for c, p in zip(top5_idxs, top5_probs)
            ],
            "probabilities": [round(p, 6) for p in probs.cpu().tolist()],
        }

    def predict_with_gradcam(
        self, image_path_or_pil: Union[str, Image.Image]
    ) -> dict:
        """
        Run inference **and** generate a Grad-CAM overlay (base64-encoded JPEG).
        """
        image = self._load_image(image_path_or_pil)

        # Grad-CAM requires gradients
        prev_grads = {}
        for name, p in self.model.named_parameters():
            prev_grads[name] = p.requires_grad
            p.requires_grad = True

        try:
            # Last conv layer in timm EfficientNetV2 is `conv_head`
            target_layer = self.model.conv_head
            grad_cam = GradCAM(self.model, target_layer)

            tensor = self.transform(image).unsqueeze(0).to(self.device)
            tensor.requires_grad = True

            # Forward + generate CAM
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            top_class = torch.argmax(probs).item()
            cam = grad_cam.generate(tensor, top_class)

            # Overlay on original image
            img_np = np.array(image.resize((256, 256)))
            if cam.ndim == 2:
                cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
                heatmap = cv2.applyColorMap(
                    np.uint8(255 * cam_resized), cv2.COLORMAP_JET
                )
                heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(
                    np.float32
                ) / 255.0
                overlay = heatmap_rgb * 0.4 + img_np.astype(np.float32) / 255.0 * 0.6
                overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
            else:
                overlay = img_np

            # Encode to base64 JPEG
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        finally:
            # Restore gradient state
            for name, p in self.model.named_parameters():
                p.requires_grad = prev_grads.get(name, False)

        result = self.predict(image)
        result["gradcam_base64"] = f"data:image/jpeg;base64,{b64}"
        return result
