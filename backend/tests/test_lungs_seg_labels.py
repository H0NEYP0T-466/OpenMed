"""Supervision scope and the display merge rule.

These two functions are where the module can quietly become wrong while every
number still looks fine:

* if the supervision flag were derived from mask content (``mask.sum() > 0``),
  Normal would be dropped along with Non-COVID and head B would train on COVID
  positives with zero negatives — a head that paints opacity everywhere and
  reports an excellent training Dice;
* if the merge rule were a softmax over three classes, background would compete
  with anatomy and every threshold downstream would be meaningless.

So both are asserted on their own, against hand-built masks, without a model.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.organs.lungs.segmentation.kaggle.catalog import Record
from app.organs.lungs.segmentation.labels import (
    CLASS_COVID,
    CLASS_NON_COVID,
    CLASS_NORMAL,
    MERGE_BACKGROUND,
    MERGE_LUNG,
    MERGE_OPACITY,
    merge_label_map,
    normalise_class,
    opacity_supervised,
)


# ─────────────────────────────────────────────────────────────────────────────
# Supervision is provenance, not pixel content
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "in_tree", "expected"),
    [
        (CLASS_COVID, True, True),
        (CLASS_NORMAL, True, True),
        # Bacterial consolidation was never delineated: an annotation negative, not a
        # true negative. Training on it teaches "consolidation is not a lesion".
        (CLASS_NON_COVID, True, False),
        # No annotation at all outside the infection tree.
        (CLASS_COVID, False, False),
        (CLASS_NORMAL, False, False),
        (CLASS_NON_COVID, False, False),
    ],
)
def test_supervision_flag_follows_provenance(name: str, in_tree: bool, expected: bool) -> None:
    assert opacity_supervised(name, in_tree) is expected


@pytest.mark.parametrize(
    "spelling", ["COVID-19", "covid-19", "Covid 19", "covid_19", "COVID19", " covid-19 "]
)
def test_supervision_flag_accepts_directory_spellings(spelling: str) -> None:
    assert opacity_supervised(spelling, True) is True


def test_supervision_ignores_mask_content() -> None:
    """The decision that matters: a *full* Non-COVID mask must still be unsupervised.

    Non-COVID masks ship all-black, but if one ever carried pixels the flag must not
    move — the flag is provenance. This is the assertion that keeps the loss honest.
    """
    def record(cls: str, in_tree: bool, infection_path: str | None) -> Record:
        return Record(
            uid=0, filename="x.png", cls=cls, shipped_split="Train",
            image_path="/nope/image.png", lung_mask_path="/nope/lung.png",
            in_infection_tree=in_tree, infection_mask_path=infection_path,
        )

    assert record(CLASS_NON_COVID, True, "/nope/full_mask.png").opacity_supervised is False
    assert record(CLASS_NORMAL, True, "/nope/black_mask.png").opacity_supervised is True
    assert record(CLASS_COVID, True, "/nope/lesion.png").opacity_supervised is True


def test_record_tree_membership_is_reported() -> None:
    inside = Record(uid=1, filename="a.png", cls=CLASS_COVID, shipped_split="Train",
                    image_path="a", lung_mask_path="b", in_infection_tree=True)
    outside = Record(uid=2, filename="b.png", cls=CLASS_COVID, shipped_split="Train",
                     image_path="a", lung_mask_path="b", in_infection_tree=False)
    assert inside.tree == "lung+infection"
    assert outside.tree == "lung_only"
    assert inside.to_dict()["opacity_supervised"] is True
    assert outside.to_dict()["opacity_supervised"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Merge rule: 0 background / 1 lung / 2 opacity, clipped to the lung
# ─────────────────────────────────────────────────────────────────────────────


def _frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lung = np.zeros((8, 8), dtype=np.uint8)
    lung[2:6, 2:6] = 255
    opacity = np.zeros((8, 8), dtype=np.uint8)
    opacity[3:5, 3:5] = 255          # inside the lung
    return lung, opacity, np.zeros((8, 8), dtype=np.uint8)


def test_merge_uses_three_values_and_only_three() -> None:
    lung, opacity, _ = _frame()
    merged = merge_label_map(lung, opacity)

    assert merged.dtype == np.uint8
    assert set(np.unique(merged).tolist()) == {MERGE_BACKGROUND, MERGE_LUNG, MERGE_OPACITY}
    assert int((merged == MERGE_LUNG).sum()) == 16 - 4
    assert int((merged == MERGE_OPACITY).sum()) == 4


def test_merge_discards_opacity_outside_the_lung() -> None:
    lung, opacity, _ = _frame()
    opacity[6:8, 6:8] = 255           # outside the lung field
    merged = merge_label_map(lung, opacity)

    assert int(merged[6, 6]) == MERGE_BACKGROUND
    # value 2 can never appear where the lung is 0
    assert not np.any((merged == MERGE_OPACITY) & (lung == 0))


def test_merge_without_an_opacity_head_is_binary() -> None:
    lung, _, _ = _frame()
    merged = merge_label_map(lung, None)
    assert set(np.unique(merged).tolist()) == {MERGE_BACKGROUND, MERGE_LUNG}


def test_merge_treats_any_positive_value_as_foreground() -> None:
    lung = np.full((4, 4), 255, dtype=np.uint8)
    opacity = np.full((4, 4), 255, dtype=np.uint8)
    merged = merge_label_map(lung, opacity)
    assert int(merged.min()) == MERGE_OPACITY
    assert int(merged.max()) == MERGE_OPACITY


# ─────────────────────────────────────────────────────────────────────────────
# Class name normalisation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("COVID-19", CLASS_COVID),
        ("COVID-19", CLASS_COVID),
        ("Non-COVID", CLASS_NON_COVID),
        ("Non_COVID", CLASS_NON_COVID),
        ("non covid", CLASS_NON_COVID),
        ("Normal", CLASS_NORMAL),
        ("normal", CLASS_NORMAL),
    ],
)
def test_normalise_class(raw: str, expected: str) -> None:
    assert normalise_class(raw) == expected


def test_normalise_class_refuses_an_unknown_directory() -> None:
    with pytest.raises(ValueError, match="Unrecognised class name"):
        normalise_class("Tuberculosis")


def test_task_is_named_for_what_it_delineates() -> None:
    """The task name travels to the API and the UI; it must not overclaim."""
    from app.organs.lungs.segmentation.labels import SCOPE_NOTICE, TASK_NAME

    assert TASK_NAME == "COVID opacity segmentation"
    assert "not a diagnosis" in SCOPE_NOTICE
