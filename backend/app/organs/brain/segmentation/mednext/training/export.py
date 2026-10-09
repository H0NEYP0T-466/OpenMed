"""Serving checkpoint format.

One file carries everything inference needs, so deployment is "drop it in":

    state_dict   weights, auxiliary deep-supervision heads removed, CPU float32
    spec         MedNeXtSpec — the exact architecture
    preprocess   the image contract (see ``preprocessor.preprocess_spec``)
    postprocess  threshold, minimum component area, flip-TTA switch, all tuned
                 on the validation split
    metrics      held-out numbers recorded at export time
    training     provenance (epoch, weight source, seed, split counts)

The file holds only tensors and plain Python types, so it loads with
``torch.load(..., weights_only=True)``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch

from ..model import MedNeXtSpec, strip_deep_supervision
from ..preprocessor import preprocess_spec

FORMAT_VERSION = 1
MODEL_KIND = "mednext_2d_brain_binary_seg"


def _default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def plain(obj: Any) -> Any:
    """Round-trip through JSON so only builtin types reach ``weights_only`` loading."""
    return json.loads(json.dumps(obj, default=_default))


def build_serving_checkpoint(
    state_dict: dict[str, torch.Tensor],
    spec: MedNeXtSpec,
    image_size: int,
    *,
    postprocess: dict[str, Any],
    metrics: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    cpu_state = {
        k: v.detach().to("cpu", dtype=torch.float32).clone()
        for k, v in strip_deep_supervision(state_dict).items()
    }
    return {
        "format_version": FORMAT_VERSION,
        "kind": MODEL_KIND,
        "state_dict": cpu_state,
        "spec": spec.to_dict(),
        "preprocess": preprocess_spec(image_size),
        "postprocess": plain(postprocess),
        "metrics": plain(metrics),
        "training": plain(training),
    }


def save_serving_checkpoint(payload: dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path
