"""Brain tumour segmentation with LiteMedSAM.

Provides a promptable segmentation pipeline backed by the distilled
MedSAM-Lite model (TinyViT-256 encoder, SAM prompt encoder + mask decoder).
Designed for seamless integration with the brain classifier: when a tumour
is detected, the Grad-CAM heatmap can be fed as a dense mask prompt.
"""
