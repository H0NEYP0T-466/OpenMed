"""Model registry for the Experimental Laboratory.

One process-wide instance per model, created on first use and dropped on
`unload`. The lab's contract with the user is that a model occupies memory only
while its tab is open, so the registry is where that promise is kept.

Adding a model: create a package with a `LabAdapter` subclass, import it at the
bottom of this file, and it appears in `/api/experimental/models` and in the
frontend's model list without touching the router.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from .base import LabAdapter

logger = logging.getLogger(__name__)

_ADAPTERS: dict[str, type[LabAdapter]] = {}
_INSTANCES: dict[str, LabAdapter] = {}
_LOCK = threading.Lock()


def register(adapter_cls: type[LabAdapter]) -> type[LabAdapter]:
    """Register an adapter class. Decorator, or call directly at import time."""
    model_id = getattr(adapter_cls, "id", "")
    if not model_id:
        raise ValueError(f"{adapter_cls.__name__} has no id")
    if model_id in _ADAPTERS and _ADAPTERS[model_id] is not adapter_cls:
        raise ValueError(f"duplicate model id {model_id!r}")
    _ADAPTERS[model_id] = adapter_cls
    return adapter_cls


def get(model_id: str) -> LabAdapter:
    """Return the singleton for `model_id`, creating it if needed.

    Creation is cheap — weights are not touched until `load()`.
    """
    with _LOCK:
        if model_id not in _ADAPTERS:
            raise KeyError(f"unknown model {model_id!r}; known: {sorted(_ADAPTERS)}")
        instance = _INSTANCES.get(model_id)
        if instance is None:
            instance = _ADAPTERS[model_id]()
            _INSTANCES[model_id] = instance
        return instance


def list_models() -> list[dict[str, Any]]:
    with _LOCK:
        classes = list(_ADAPTERS.values())
    out = []
    for cls in classes:
        try:
            out.append(get(cls.id).info())
        except Exception as exc:  # noqa: BLE001 - one bad adapter must not break the list
            logger.warning("model %s failed to report: %s", cls.id, exc)
            out.append({"id": cls.id, "title": cls.title, "available": False,
                        "unavailable_reason": str(exc), "loaded": False})
    return sorted(out, key=lambda m: m["id"])


def unload(model_id: str) -> bool:
    with _LOCK:
        instance = _INSTANCES.get(model_id)
    if instance is None:
        return False
    instance.unload()
    logger.info("unloaded %s", model_id)
    return True


def unload_all() -> list[str]:
    with _LOCK:
        ids = list(_INSTANCES)
    return [model_id for model_id in ids if unload(model_id)]


def loaded_models() -> list[str]:
    with _LOCK:
        instances = list(_INSTANCES.items())
    return [model_id for model_id, adapter in instances if adapter.is_loaded]


# ── registrations ────────────────────────────────────────────────────────────
# Import after the helpers above to avoid a circular import at module load.

from .medsam2.adapter import MedSAM2Adapter  # noqa: E402

register(MedSAM2Adapter)

__all__ = ["register", "get", "list_models", "unload", "unload_all", "loaded_models"]
