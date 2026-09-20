"""Transform resolution for training, validation and inference.

All three are derived from the same timm model data config so that inference
preprocessing cannot drift away from what the network was trained on.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from timm.data import create_transform, resolve_model_data_config

logger = logging.getLogger(__name__)


def data_config(model: Any) -> dict[str, Any]:
    return dict(resolve_model_data_config(model))


def get_train_transform(model: Any, auto_augment: Optional[str] = None) -> Any:
    config = data_config(model)
    if auto_augment:
        config["auto_augment"] = auto_augment
    logger.info("Resolved train transform config: %s", config)
    return create_transform(**config, is_training=True)


def get_minority_train_transform(
    model: Any,
    auto_augment: str = "rand-m7-mstd0.5-inc1",
) -> Any:
    """Stronger augmentation stack for lower-sample classes to boost effective diversity."""
    config = data_config(model)
    config["auto_augment"] = auto_augment
    config["scale"] = (0.7, 1.0)
    config["hflip"] = 0.5
    config["vflip"] = 0.2
    logger.info("Resolved minority train transform config: %s", config)
    return create_transform(**config, is_training=True)


def build_class_aware_transforms(
    model: Any,
    class_names: Sequence[str],
    minority_classes: Sequence[str],
    base_augment: Optional[str] = None,
    minority_augment: str = "rand-m7-mstd0.5-inc1",
) -> tuple[Any, dict[str, Any]]:
    """Return (default_train_transform, dict_of_transforms_for_minority_classes)."""
    default_transform = get_train_transform(model, auto_augment=base_augment)
    minority_transform = get_minority_train_transform(model, auto_augment=minority_augment)
    class_transforms = {
        cls_name: minority_transform for cls_name in minority_classes if cls_name in class_names
    }
    return default_transform, class_transforms



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
