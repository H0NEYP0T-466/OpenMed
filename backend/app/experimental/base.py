"""Contract every Experimental Laboratory model implements.

The lab is a place to compare promptable segmentation models side by side. Each
model lives in its own package and exposes one adapter; the registry finds them
and the router serves them through a single set of endpoints.

Adding a model means adding a package with an adapter subclass and importing it
in `registry.py`. Nothing else changes — not the router, not the frontend
contract.

Why an adapter and not a bare function: every model here has to be *loaded*,
*unloaded*, and asked whether it is available before it is loaded. Those three
answers are the difference between a lab that frees memory and one that leaks a
gigabyte per tab visit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from PIL import Image


@dataclass(frozen=True)
class PointPrompt:
    """One click. `positive=False` marks background — SAM2's negative prompt."""

    x: int
    y: int
    positive: bool = True


@dataclass(frozen=True)
class SegPrompt:
    """Any combination of the three prompt types SAM2 understands.

    Box, points and a mask are not alternatives — they compose. A box with a
    negative point inside it is a legitimate prompt, and a mask with two
    correction clicks is how a user fixes a bad first pass.

    Coordinates are original-image pixels, un-normalised: every model wants
    pixels at its own resolution and converting once at the boundary is how
    off-by-scale bugs start.
    """

    box: Optional[tuple[int, int, int, int]] = None
    points: tuple[PointPrompt, ...] = ()
    mask_png: Optional[bytes] = None

    def __post_init__(self) -> None:
        if self.box is None and not self.points and self.mask_png is None:
            raise ValueError("a prompt needs at least one of: box, points, mask")
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"degenerate box: {self.box}")
            if x1 < 0 or y1 < 0:
                raise ValueError(f"negative box origin: {self.box}")

    @property
    def kinds(self) -> list[str]:
        kinds = []
        if self.box is not None:
            kinds.append("box")
        if self.points:
            kinds.append("points")
        if self.mask_png is not None:
            kinds.append("mask")
        return kinds

    def clamped_box(self, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
        if self.box is None:
            return None
        x1, y1, x2, y2 = self.box
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        return (x1, y1, x2, y2)

    def clamped_points(self, width: int, height: int) -> tuple[PointPrompt, ...]:
        return tuple(
            PointPrompt(max(0, min(p.x, width - 1)), max(0, min(p.y, height - 1)), p.positive)
            for p in self.points
        )


@dataclass
class MaskResult:
    """What every adapter returns, so the UI renders all models the same way."""

    mask_base64: str
    overlay_base64: str
    score: float
    inference_ms: float
    image_size: tuple[int, int]
    model: str
    extra: dict[str, Any] = field(default_factory=dict)


class LabAdapter(ABC):
    """One promptable model, loaded on demand and released on request."""

    #: stable identifier used in URLs and the frontend registry
    id: str = ""
    #: display name
    title: str = ""
    #: one line for the tab card
    description: str = ""
    #: native working resolution, for the UI's "expected input" note
    input_size: int = 0

    @abstractmethod
    def availability(self) -> tuple[bool, str]:
        """(can_load, reason). Called by /health before anything is loaded."""

    @abstractmethod
    def load(self) -> float:
        """Load weights, return milliseconds spent. Must be idempotent."""

    @abstractmethod
    def unload(self) -> None:
        """Release the model and free device memory. Must be idempotent."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def predict(self, image: Image.Image, prompt: SegPrompt) -> MaskResult:
        """Run one inference. May lazily load; the router handles the lock."""

    def info(self) -> dict[str, Any]:
        """Metadata for /models and the docs panel."""
        available, reason = self.availability()
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "input_size": self.input_size,
            "available": available,
            "unavailable_reason": None if available else reason,
            "loaded": self.is_loaded,
        }


def resolve_device(preference: Optional[str] = None) -> str:
    """cpu or cuda, with an explicit override for debugging."""
    import torch

    if preference:
        return preference
    return "cuda" if torch.cuda.is_available() else "cpu"
