"""Self-describing serving checkpoint, and the strict loader.

One ``.pth`` carries everything inference needs, so deployment is "drop it in":

    state_dict    canonical-named weights, CPU float32 (see ``model.py``)
    spec          the exact architecture, including which heads exist
    preprocess    the image contract (``preprocessor.preprocess_spec``)
    postprocess   tuned lung and opacity post-processing
    labels        task name, heads, heads actually trained, display merge legend
    metrics       held-out numbers recorded at export time
    training      provenance (epoch, weight source, seed, split counts)

The file holds only tensors and plain Python types, so it loads with
``torch.load(..., weights_only=True)``. A JSON sidecar written beside it repeats
everything except the weights, so metrics and the label space can be read (and
shown in the UI) without touching the network.

Loading fails closed. A checkpoint whose preprocessing version, head set or
architecture does not match this build is refused with a precise reason and is
never resized or coerced to fit.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np
import torch

from .labels import HEADS, LEGEND, TASK_NAME
from .model import ENCODER_DEBUG, SegSpec
from .preprocessor import PREPROCESS_VERSION, preprocess_spec

FORMAT_VERSION = 1
MODEL_KIND = "lungs_segformer_two_head_seg"
CHECKPOINT_NAME = "lungs_seg.pth"
SIDECAR_NAME = "lungs_seg.sidecar.json"
CHECKPOINT_ENV = "OPENMED_LUNGS_SEG_CHECKPOINT"
ALLOW_DEBUG_ENV = "OPENMED_LUNGS_SEG_ALLOW_DEBUG"


class SegmentationUnavailableError(RuntimeError):
    """The checkpoint is missing or unusable. Serving fails closed with this reason.

    ``code`` is the typed error the HTTP layer reports: ``checkpoint_missing`` or
    ``checkpoint_unusable``.
    """

    def __init__(self, message: str, code: str = "checkpoint_unusable") -> None:
        super().__init__(message)
        self.code = code


def checkpoints_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def default_checkpoint_path() -> str:
    configured = os.getenv(CHECKPOINT_ENV)
    if configured:
        return os.path.expanduser(configured)
    return os.path.join(checkpoints_dir(), CHECKPOINT_NAME)


def sidecar_path_for(checkpoint_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), SIDECAR_NAME)


# ------------------------------------------------------------------------- JSON safety


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _scrub(obj: Any) -> Any:
    """Replace non-finite floats with ``None``: a NaN must not look like a measurement."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return obj


def plain(obj: Any) -> Any:
    """Round-trip through JSON so only builtin types reach ``weights_only`` loading."""
    return _scrub(json.loads(json.dumps(obj, default=_json_default)))


# --------------------------------------------------------------------------- building


def labels_block(spec: SegSpec, heads_trained: tuple[str, ...]) -> dict[str, Any]:
    return {
        "task": TASK_NAME,
        "heads": list(spec.heads),
        "heads_trained": list(heads_trained),
        "merge_legend": [dict(item) for item in LEGEND],
        "merge_rule": "0 background, 1 lung field only, 2 opacity clipped to the lung field",
    }


def build_serving_checkpoint(
    state_dict: dict[str, torch.Tensor],
    spec: SegSpec,
    *,
    heads_trained: tuple[str, ...],
    postprocess: dict[str, Any],
    metrics: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    if not set(heads_trained) <= set(spec.heads):
        raise ValueError(f"heads_trained {heads_trained} is not a subset of {spec.heads}.")
    cpu_state = {
        key: value.detach().to("cpu", dtype=torch.float32).clone()
        for key, value in state_dict.items()
    }
    return {
        "format_version": FORMAT_VERSION,
        "kind": MODEL_KIND,
        "state_dict": cpu_state,
        "spec": plain(spec.to_dict()),
        "preprocess": plain(preprocess_spec()),
        "postprocess": plain(postprocess),
        "labels": plain(labels_block(spec, heads_trained)),
        "metrics": plain(metrics),
        "training": plain(training),
    }


def sidecar_from_payload(payload: dict[str, Any], checkpoint_name: str) -> dict[str, Any]:
    """Everything except the weights."""
    return {
        "checkpoint": checkpoint_name,
        "format_version": payload["format_version"],
        "kind": payload["kind"],
        "label_space": payload["labels"],
        "architecture": payload["spec"],
        "preprocess": payload["preprocess"],
        "preprocess_version": payload["preprocess"]["version"],
        "postprocess": payload["postprocess"],
        "metrics": payload["metrics"],
        "training": payload["training"],
    }


def save_serving_checkpoint(payload: dict[str, Any], path: str) -> tuple[str, str]:
    """Write the ``.pth`` and its sidecar atomically. Returns both paths."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    sidecar = sidecar_path_for(path)
    sidecar_tmp = sidecar + ".tmp"
    with open(sidecar_tmp, "w", encoding="utf-8") as handle:
        json.dump(sidecar_from_payload(payload, os.path.basename(path)), handle, indent=2)
        handle.write("\n")
    os.replace(sidecar_tmp, sidecar)
    return path, sidecar


# --------------------------------------------------------------------------- loading


def load_serving_checkpoint(path: str) -> dict[str, Any]:
    """Read and validate a serving checkpoint without building the network."""
    if not os.path.isfile(path):
        raise SegmentationUnavailableError(
            f"Lungs segmentation checkpoint not found at {path}. Train it on Kaggle with "
            f"kaggle/train_lungs_seg.py and place {CHECKPOINT_NAME} (and its sidecar "
            f"{SIDECAR_NAME}) in that folder.",
            code="checkpoint_missing",
        )
    if os.path.getsize(path) == 0:
        raise SegmentationUnavailableError(f"Lungs segmentation checkpoint at {path} is empty.")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - any unreadable file is fail-closed
        raise SegmentationUnavailableError(
            f"Lungs segmentation checkpoint is unreadable: {exc}"
        ) from exc

    if not isinstance(payload, dict) or payload.get("kind") != MODEL_KIND:
        raise SegmentationUnavailableError(
            "File is not a lungs segmentation serving checkpoint "
            "(use best/lungs_seg.pth from the run, not last.pt)."
        )
    if payload.get("format_version") != FORMAT_VERSION:
        raise SegmentationUnavailableError(
            f"Unsupported checkpoint format {payload.get('format_version')!r}; "
            f"this build reads format {FORMAT_VERSION}."
        )
    for key in ("state_dict", "spec", "preprocess", "postprocess", "labels"):
        if key not in payload:
            raise SegmentationUnavailableError(f"Checkpoint is missing the {key!r} section.")

    version = payload["preprocess"].get("version")
    if version != PREPROCESS_VERSION:
        raise SegmentationUnavailableError(
            f"Checkpoint was trained with preprocessing v{version}; this build implements "
            f"v{PREPROCESS_VERSION}. Serving it would feed the network inputs it never saw."
        )

    try:
        spec = SegSpec.from_dict(payload["spec"])
    except (TypeError, ValueError) as exc:
        raise SegmentationUnavailableError(
            f"Checkpoint declares an architecture this build cannot construct: {exc}"
        ) from exc

    declared = tuple(payload["labels"].get("heads", ()))
    if declared != spec.heads:
        raise SegmentationUnavailableError(
            f"Checkpoint label space declares heads {list(declared)} but its architecture has "
            f"{list(spec.heads)}."
        )
    unknown = [h for h in declared if h not in HEADS]
    if unknown:
        raise SegmentationUnavailableError(f"Checkpoint declares unknown heads {unknown}.")
    trained = tuple(payload["labels"].get("heads_trained", ()))
    if "lung" not in trained or not set(trained) <= set(spec.heads):
        raise SegmentationUnavailableError(
            f"Checkpoint reports heads_trained={list(trained)}; the lung head must be trained and "
            f"trained heads must be a subset of {list(spec.heads)}."
        )
    if spec.encoder == ENCODER_DEBUG and os.getenv(ALLOW_DEBUG_ENV) != "1":
        raise SegmentationUnavailableError(
            "Checkpoint was built on the debug encoder (a CPU smoke-test network, not a result); "
            "refusing to serve it."
        )
    return payload


def read_sidecar(checkpoint_path: str) -> dict[str, Any] | None:
    """The JSON sidecar beside the checkpoint, or ``None`` when absent or malformed."""
    path = sidecar_path_for(checkpoint_path)
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
