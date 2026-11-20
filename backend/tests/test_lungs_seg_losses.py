"""The two-head loss.

Two invariants from the brief that are easy to lose in a refactor:

* Dice is computed **per sample** and then averaged. A batch-averaged Dice lets a
  130 px lesion disappear inside 256x256 lungs, which is exactly the failure the
  crop exists to prevent.
* Head B's term only counts genuinely supervised samples. The dummy crop that
  keeps DDP's graph complete must contribute zero, or the head learns from a
  frame it was never given a label for.
"""

from __future__ import annotations

import pytest
import torch

from app.organs.lungs.segmentation.kaggle.losses import (
    TwoHeadLoss,
    per_sample_bce,
    per_sample_dice_loss,
)


def _loss() -> TwoHeadLoss:
    return TwoHeadLoss(lung_weight=1.0, opacity_weight=1.0, bce_weight=0.5,
                       dice_weight=0.5, dice_eps=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# per-sample Dice
# ─────────────────────────────────────────────────────────────────────────────


def test_dice_is_per_sample_so_a_tiny_lesion_still_counts() -> None:
    """A confident small prediction and a missed large one must not cancel out."""
    good_small = torch.full((8, 8), -6.0)         # predicts nothing...
    good_small[3:5, 3:5] = 6.0                    # ...except the tiny lesion, exactly
    target_small = torch.zeros((8, 8))
    target_small[3:5, 3:5] = 1.0

    missed_large = torch.full((8, 8), -6.0)       # predicts nothing at all
    target_large = torch.zeros((8, 8))
    target_large[:6, :6] = 1.0

    logits = torch.stack([good_small, missed_large])
    targets = torch.stack([target_small, target_large])
    per_sample = per_sample_dice_loss(logits, targets, eps=1.0)

    assert per_sample.shape == (2,)
    assert float(per_sample[0]) < 0.3            # the small correct one is rewarded
    assert float(per_sample[1]) > 0.7            # the missed large one is punished
    # averaging the two masks first would report one blended, meaningless number
    blended = per_sample_dice_loss(logits.mean(0, keepdim=True), targets.mean(0, keepdim=True), 1.0)
    assert float(blended[0]) != pytest.approx(float(per_sample.mean()), abs=1e-4)


def test_dice_is_bounded_and_symmetric_in_shape() -> None:
    logits = torch.randn(3, 16, 16)
    targets = (torch.rand(3, 16, 16) > 0.5).float()
    values = per_sample_dice_loss(logits, targets, eps=1.0)
    assert values.shape == (3,)
    assert bool(torch.all(values >= 0.0)) and bool(torch.all(values <= 1.0))
    assert bool(torch.all(torch.isfinite(values)))


def test_empty_target_and_empty_prediction_is_not_nan() -> None:
    logits = torch.full((1, 8, 8), -20.0)
    target = torch.zeros((1, 8, 8))
    value = per_sample_dice_loss(logits, target, eps=1.0)
    assert torch.isfinite(value).all()


def test_bce_is_reduced_per_sample_not_over_the_batch() -> None:
    logits = torch.full((4, 8, 8), -3.0)         # confidently background
    targets = torch.zeros(4, 8, 8)
    logits[0] = -8.0                              # confidently wrong about sample 0
    targets[0] = 1.0
    values = per_sample_bce(logits, targets)

    assert values.shape == (4,)
    assert float(values[0]) > 5.0                 # punished alone
    assert float(values[1]) < 0.2                 # rewarded alone


# ─────────────────────────────────────────────────────────────────────────────
# the two-head assembly
# ─────────────────────────────────────────────────────────────────────────────


def _batch(supervised: int = 2, crops: int = 2):
    lungs = torch.randn(4, 16, 16)
    lung_targets = (torch.rand(4, 16, 16) > 0.6).float()
    op_logits = torch.randn(crops, 16, 16)
    op_targets = (torch.rand(crops, 16, 16) > 0.8).float()
    valid = torch.zeros(crops)
    valid[:supervised] = 1.0
    return lungs, lung_targets, op_logits, op_targets, valid


def test_lung_only_stage_has_no_opacity_term() -> None:
    lungs, lung_targets, _, _, _ = _batch()
    total, parts = _loss()(lungs, lung_targets)
    assert float(parts["opacity_total"]) == 0.0
    assert float(parts["n_supervised"]) == 0.0
    assert float(total) > 0.0
    assert total.detach().requires_grad is False


def test_dummy_crop_contributes_nothing() -> None:
    """The zero-weight crop exists only to keep DDP's reducer complete."""
    lungs, lung_targets, op_logits, op_targets, _ = _batch()
    real, _ = _loss()(lungs, lung_targets, op_logits[:1], op_targets[:1],
                      torch.ones(1))
    with_dummy, _ = _loss()(lungs, lung_targets, op_logits, op_targets,
                            torch.tensor([1.0, 0.0]))

    assert float(with_dummy) == pytest.approx(float(real), abs=1e-6)


def test_a_zero_weight_crop_does_not_move_the_opacity_term() -> None:
    """Only supervised crops count: weighting crop 2 to zero must equal not passing it."""
    lungs, lung_targets, op_logits, op_targets, _ = _batch()
    mixed, mixed_parts = _loss()(
        lungs, lung_targets, op_logits, op_targets, torch.tensor([1.0, 0.0]))
    single, single_parts = _loss()(
        lungs, lung_targets, op_logits[:1], op_targets[:1], torch.ones(1))

    assert float(mixed_parts["n_supervised"]) == 1.0
    assert float(mixed) == pytest.approx(float(single), abs=1e-6)
    assert float(mixed_parts["opacity_bce"]) == pytest.approx(
        float(single_parts["opacity_bce"]), abs=1e-6)

    # and both crops together really do give a different number
    both, both_parts = _loss()(lungs, lung_targets, op_logits, op_targets, torch.ones(2))
    assert float(both_parts["n_supervised"]) == 2.0
    assert float(both_parts["opacity_bce"]) != pytest.approx(
        float(single_parts["opacity_bce"]), abs=1e-6)


def test_total_is_the_weighted_sum_of_its_parts() -> None:
    lungs, lung_targets, op_logits, op_targets, valid = _batch()
    total, parts = _loss()(lungs, lung_targets, op_logits, op_targets, valid)
    expected = (parts["lung_total"] + parts["opacity_total"]) / 1.0
    assert float(total) == pytest.approx(float(expected), abs=1e-6)


def test_gradients_reach_both_heads() -> None:
    lungs = torch.randn(4, 16, 16, requires_grad=True)
    op_logits = torch.randn(2, 16, 16, requires_grad=True)
    lung_targets = (torch.rand(4, 16, 16) > 0.6).float()
    op_targets = (torch.rand(2, 16, 16) > 0.8).float()

    total, _ = _loss()(lungs, lung_targets, op_logits, op_targets, torch.ones(2))
    total.backward()

    assert lungs.grad is not None and float(lungs.grad.abs().sum()) > 0.0
    assert op_logits.grad is not None and float(op_logits.grad.abs().sum()) > 0.0
