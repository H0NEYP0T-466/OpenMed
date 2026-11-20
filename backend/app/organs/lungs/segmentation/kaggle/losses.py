"""Two-head loss: per-sample soft Dice plus BCE-with-logits, per head.

Dice is computed **per sample** and then averaged. A batch-averaged Dice would let
the large lung fields drown a small opacity; a 130 px lesion is 0.2 % of the frame
and disappears inside a pooled sum.

Total = ``lung_weight * lung_term`` over every sample
      + ``opacity_weight * opacity_term`` over supervised samples only.

Supervised samples are selected by the provenance flag from ``labels.py`` (COVID-19
positives and Normal true negatives of the infection tree), never by mask content.
Normal samples have an empty target, and the per-sample Dice with ``eps`` smoothing
turns that into a bounded penalty on any predicted mass: that is what stops head B
hallucinating opacity on clean lungs. Non-COVID never reaches this function.

Everything is computed in float32 whatever the autocast dtype. Per-term scalars are
returned for logging; there is no single opaque number.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def per_sample_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    """``1 - soft Dice`` for each sample; ``logits`` and ``target`` are ``(B, H, W)``."""
    prob = torch.sigmoid(logits.float())
    target = target.float()
    inter = (prob * target).sum(dim=(1, 2))
    denom = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def per_sample_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction="none"
    ).mean(dim=(1, 2))


class TwoHeadLoss:
    def __init__(self, *, lung_weight: float, opacity_weight: float, bce_weight: float,
                 dice_weight: float, dice_eps: float) -> None:
        self.lung_weight = lung_weight
        self.opacity_weight = opacity_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_eps = dice_eps

    def head(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample ``(bce, dice_loss)``."""
        return per_sample_bce(logits, target), per_sample_dice_loss(logits, target, self.dice_eps)

    def combine(self, bce: torch.Tensor, dice: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * bce + self.dice_weight * dice

    def __call__(
        self,
        lung_logits: torch.Tensor,
        lung_target: torch.Tensor,
        opacity_logits: torch.Tensor | None = None,
        opacity_target: torch.Tensor | None = None,
        opacity_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``opacity_valid`` is a ``(n_crops,)`` 0/1 weight. It is all-ones for real
        supervised crops and zero for the dummy crop that keeps DDP's graph complete
        when a batch happens to hold no supervised sample."""
        lung_bce, lung_dice = self.head(lung_logits, lung_target)
        lung_term = self.combine(lung_bce, lung_dice).mean()
        total = self.lung_weight * lung_term
        zero = lung_term.detach() * 0.0
        parts: dict[str, torch.Tensor] = {
            "lung_bce": lung_bce.mean().detach(), "lung_dice_loss": lung_dice.mean().detach(),
            "lung_total": lung_term.detach(), "opacity_bce": zero, "opacity_dice_loss": zero,
            "opacity_total": zero, "n_supervised": zero,
        }
        if opacity_logits is not None and opacity_target is not None:
            valid = (opacity_valid if opacity_valid is not None
                     else torch.ones(opacity_logits.shape[0], device=opacity_logits.device)).float()
            op_bce, op_dice = self.head(opacity_logits, opacity_target)
            op_per = self.combine(op_bce, op_dice)
            denom = valid.sum().clamp(min=1.0)
            op_term = (op_per * valid).sum() / denom
            total = total + self.opacity_weight * op_term
            parts.update({
                "opacity_bce": ((op_bce * valid).sum() / denom).detach(),
                "opacity_dice_loss": ((op_dice * valid).sum() / denom).detach(),
                "opacity_total": op_term.detach(), "n_supervised": valid.sum().detach(),
            })
        parts["total"] = total.detach()
        return total, parts
