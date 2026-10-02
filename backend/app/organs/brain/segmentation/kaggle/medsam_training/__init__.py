"""Modular LiteMedSAM fine-tuning package for Kaggle.

Entry point is ``kaggle/trainKaggle.py``. Every stage is importable and
runnable on its own so a Kaggle session can be debugged stage by stage:

    config      — hyper-parameters and path resolution
    bootstrap   — environment report, dependency check, weight download/verify
    audit       — dataset hygiene (pairing, duplicates, geometry, alignment)
    data        — split construction, Dataset, transforms, collate
    engine      — DDP training loop, evaluation, metrics, early stopping
    reporting   — plots, logs, summaries, ZIP export
    medsam_model / preprocessor — frozen snapshots of the serving code

The two snapshot modules are byte-identical to the serving implementation in
``backend/app/organs/brain/segmentation/`` so the fine-tuned weights are
drop-in compatible with the web inference path.
"""

__version__ = "1.0.0"

__all__ = [
    "config",
    "bootstrap",
    "audit",
    "data",
    "engine",
    "reporting",
]
