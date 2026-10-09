"""Loss: BCE + soft Dice, with deep supervision.

Foreground is 0.06-17 % of the frame (median ~1.5 %), so plain BCE collapses to
"predict nothing". Dice is computed per sample, which matches how the metric is
reported and stops large lesions from drowning small ones. Everything is
evaluated in float32 regardless of autocast.

Deep supervision: each auxiliary head is scored against the mask area-averaged
down to that head's resolution, with weight ``decay**i`` normalised to sum to 1.
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits.float())
    target = target.float()
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * target).sum(dims)
    total = probs.sum(dims) + target.sum(dims)
    dice = (2.0 * intersection + smooth) / (total + smooth)
    return 1.0 - dice.mean()


class CompoundLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, ds_decay: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.ds_decay = ds_decay

    def single(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target.float())
        dice = soft_dice_loss(logits, target)
        return bce, dice

    def forward(
        self,
        outputs: Union[torch.Tensor, list[torch.Tensor]],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        heads = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        weights = [self.ds_decay**i for i in range(len(heads))]
        norm = sum(weights)

        total = target.new_zeros((), dtype=torch.float32)
        parts: dict[str, torch.Tensor] = {}
        for i, (head, weight) in enumerate(zip(heads, weights, strict=True)):
            tgt = target if head.shape[-2:] == target.shape[-2:] else F.interpolate(
                target.float(), size=head.shape[-2:], mode="area"
            )
            bce, dice = self.single(head, tgt)
            total = total + (weight / norm) * (self.bce_weight * bce + self.dice_weight * dice)
            if i == 0:
                parts = {"bce": bce.detach(), "dice_loss": dice.detach()}
        parts["total"] = total.detach()
        return total, parts
