"""TorchXRayVision model loading with a persistent local weight cache.

The published DenseNet-121 checkpoint (`densenet121-res224-all`) is trained on
NIH, CheXpert, MIMIC-CXR, PadChest, and VinDr across 18 findings. Weights are
resolved from `checkpoints/` first; the network is touched only when no local
copy exists, and whatever is downloaded is copied back into that folder so the
next session never re-downloads.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
from typing import Any, Optional

logger = logging.getLogger(__name__)

WEIGHTS_NAME = "densenet121-res224-all"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

# Where torchxrayvision parks downloads when it is allowed to fetch.
_XRV_CACHE_DIRS = (
    os.path.join(os.path.expanduser("~"), ".torchxrayvision", "models_data"),
    os.path.join(os.path.expanduser("~"), ".cache", "torchxrayvision"),
)


def weights_path() -> str:
    return os.path.join(CHECKPOINT_DIR, f"{WEIGHTS_NAME}.pt")


def weights_present() -> bool:
    return os.path.isfile(weights_path())


def _persist_download() -> Optional[str]:
    """Copy a freshly downloaded checkpoint into our own folder."""
    target = weights_path()
    for cache_dir in _XRV_CACHE_DIRS:
        for candidate in glob.glob(os.path.join(cache_dir, "*.pt")):
            if WEIGHTS_NAME in os.path.basename(candidate):
                try:
                    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                    shutil.copy2(candidate, target)
                    logger.info("Persisted TorchXRayVision weights to %s", target)
                    return target
                except OSError as exc:
                    logger.warning("Could not persist weights: %s", exc)
    return None


def load_model() -> Any:
    """Build the DenseNet-121 baseline. Raises ImportError when xrv is missing."""
    import torchxrayvision as xrv  # noqa: PLC0415 - optional dependency by design

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    local = weights_path()

    if os.path.isfile(local):
        logger.info("Loading TorchXRayVision weights from %s", local)
        model = xrv.models.DenseNet(weights=local)
    else:
        logger.info("No local weights at %s — downloading %s once", local, WEIGHTS_NAME)
        model = xrv.models.DenseNet(weights=WEIGHTS_NAME)
        _persist_download()

    model.eval()
    return model


def pathology_names(model: Any) -> list[str]:
    names = getattr(model, "pathologies", None) or []
    return [str(name) for name in names]
