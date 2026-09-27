"""Brain tumour segmentation with LiteMedSAM.

Provides a promptable segmentation pipeline backed by the distilled
MedSAM-Lite model (TinyViT-256 encoder, SAM prompt encoder + mask decoder).
Designed for seamless integration with the brain classifier: when a tumour is
detected, a bounding box derived from the Grad-CAM heatmap is used as the
segmentation prompt. The released LiteMedSAM weights were trained to segment
from boxes only, so the dense mask prompt path is not used.
"""
