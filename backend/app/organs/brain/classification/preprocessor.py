"""Transform resolution for training, validation and inference.

All three are derived from the same timm model data config so that inference
preprocessing cannot drift away from what the network was trained on.
"""

from __future__ import annotations

import logging
from typing import Any

from timm.data import create_transform, resolve_model_data_config

logger = logging.getLogger(__name__)


def data_config(model: Any) -> dict[str, Any]:
    return dict(resolve_model_data_config(model))


def get_train_transform(model: Any) -> Any:
    config = data_config(model)
    logger.info("Resolved train transform config: %s", config)
    return create_transform(**config, is_training=True)


def get_val_transform(model: Any) -> Any:
    config = data_config(model)
    logger.info("Resolved val transform config: %s", config)
    return create_transform(**config, is_training=False)


def get_inference_transform(model: Any) -> Any:
    return get_val_transform(model)


def input_side_length(model: Any) -> int:
    """Spatial side length the model expects after preprocessing, in pixels."""
    size = data_config(model).get("input_size", (3, 224, 224))
    return int(size[-1])
