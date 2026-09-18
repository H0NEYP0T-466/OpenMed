import timm
from timm.data import resolve_model_data_config, create_transform
import logging

logger = logging.getLogger(__name__)

def get_train_transform(model):
    """Returns training transforms derived from model config."""
    config = resolve_model_data_config(model)
    transform = create_transform(**config, is_training=True)
    logger.info(f"Resolved Train Transform Config: {config}")
    return transform

def get_val_transform(model):
    """Returns validation transforms derived from model config."""
    config = resolve_model_data_config(model)
    transform = create_transform(**config, is_training=False)
    logger.info(f"Resolved Val Transform Config: {config}")
    return transform

def get_inference_transform(model):
    """Returns inference transforms derived from model config."""
    return get_val_transform(model)
