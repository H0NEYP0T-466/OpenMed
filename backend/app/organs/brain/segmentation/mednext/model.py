"""MedNeXt-2D — architecture shared by training and serving.

A 2D re-implementation of MedNeXt v1 (Roy et al., MICCAI 2023,
https://arxiv.org/abs/2303.16670; reference code MIC-DKFZ/MedNeXt, Apache-2.0).
The block layout, the S/B/M/L stage configurations and the up/down-sampling
blocks follow the reference; the 3D convolutions are swapped for 2D ones.

Block (all residual):
    depthwise k x k conv -> GroupNorm(groups = channels) -> 1x1 expand (x R)
    -> GELU -> [GRN] -> 1x1 compress

Network: stem, four encoder stages, a bottleneck, four decoder stages with
additive skip connections, and deep-supervision heads on the four coarser
decoder outputs. Input side must be a multiple of 16.

This file is the single source of truth for the architecture. A checkpoint
embeds the :class:`MedNeXtSpec` it was trained with, so serving rebuilds the
identical network without any hand-kept configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DOWNSAMPLE_FACTOR = 16

VARIANTS: dict[str, dict[str, Any]] = {
    "S": {"exp_r": (2,) * 9, "block_counts": (2,) * 9},
    "B": {"exp_r": (2, 3, 4, 4, 4, 4, 4, 3, 2), "block_counts": (2,) * 9},
    "M": {"exp_r": (2, 3, 4, 4, 4, 4, 4, 3, 2), "block_counts": (3, 4, 4, 4, 4, 4, 4, 4, 3)},
    "L": {"exp_r": (3, 4, 8, 8, 8, 8, 8, 4, 3), "block_counts": (3, 4, 8, 8, 8, 8, 8, 4, 3)},
}


@dataclass(frozen=True)
class MedNeXtSpec:
    """Everything needed to rebuild the network. Serialised into checkpoints."""

    variant: str = "B"
    in_channels: int = 1
    n_classes: int = 1
    n_channels: int = 32
    kernel_size: int = 3
    exp_r: tuple[int, ...] = VARIANTS["B"]["exp_r"]
    block_counts: tuple[int, ...] = VARIANTS["B"]["block_counts"]
    grn: bool = False
    drop_path_rate: float = 0.0

    def __post_init__(self) -> None:
        if len(self.exp_r) != 9 or len(self.block_counts) != 9:
            raise ValueError("exp_r and block_counts must each have 9 entries.")
        if self.kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd (the up-sampling shape fix assumes it).")
        if self.n_channels < 1:
            raise ValueError("n_channels must be a positive integer.")

    @classmethod
    def from_variant(cls, variant: str, **overrides: Any) -> MedNeXtSpec:
        key = variant.upper()
        if key not in VARIANTS:
            raise ValueError(f"Unknown MedNeXt variant {variant!r}; choose from {sorted(VARIANTS)}.")
        params: dict[str, Any] = {"variant": key, **VARIANTS[key]}
        params.update(overrides)
        return cls(**params)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["exp_r"] = list(self.exp_r)
        payload["block_counts"] = list(self.block_counts)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MedNeXtSpec:
        data = dict(payload)
        data["exp_r"] = tuple(int(v) for v in data["exp_r"])
        data["block_counts"] = tuple(int(v) for v in data["block_counts"])
        return cls(**data)


class DropPath(nn.Module):
    """Stochastic depth: zero a whole residual branch per sample."""

    def __init__(self, rate: float) -> None:
        super().__init__()
        self.rate = float(rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rate == 0.0 or not self.training:
            return x
        keep = 1.0 - self.rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class GRN(nn.Module):
    """Global Response Normalisation (ConvNeXt-V2), computed in float32.

    The L2 norm sums squares over the whole feature map; in fp16 that overflows
    on 256x256 inputs, so the statistic is accumulated in float32.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.linalg.vector_norm(x, ord=2, dim=(-2, -1), keepdim=True, dtype=torch.float32)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        nx = nx.to(x.dtype)
        return self.gamma.to(x.dtype) * (x * nx) + self.beta.to(x.dtype) + x


class MedNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int,
        kernel_size: int,
        do_res: bool = True,
        grn: bool = False,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.do_res = do_res
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride=1,
            padding=kernel_size // 2, groups=in_channels,
        )
        self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        self.conv2 = nn.Conv2d(in_channels, exp_r * in_channels, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN(exp_r * in_channels) if grn else None
        self.conv3 = nn.Conv2d(exp_r * in_channels, out_channels, kernel_size=1)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act(self.conv2(self.norm(y)))
        if self.grn is not None:
            y = self.grn(y)
        y = self.conv3(y)
        if self.do_res:
            y = x + self.drop_path(y)
        return y


class MedNeXtDownBlock(MedNeXtBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int,
        kernel_size: int,
        do_res: bool = True,
        grn: bool = False,
    ) -> None:
        super().__init__(in_channels, out_channels, exp_r, kernel_size, do_res=False, grn=grn)
        self.resample_do_res = do_res
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2) if do_res else None
        )
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride=2,
            padding=kernel_size // 2, groups=in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self.res_conv is not None:
            y = y + self.res_conv(x)
        return y


class MedNeXtUpBlock(MedNeXtBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int,
        kernel_size: int,
        do_res: bool = True,
        grn: bool = False,
    ) -> None:
        super().__init__(in_channels, out_channels, exp_r, kernel_size, do_res=False, grn=grn)
        self.resample_do_res = do_res
        self.res_conv = (
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=2) if do_res else None
        )
        self.conv1 = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size, stride=2,
            padding=kernel_size // 2, groups=in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(super().forward(x), (1, 0, 1, 0))
        if self.res_conv is not None:
            y = y + F.pad(self.res_conv(x), (1, 0, 1, 0))
        return y


class MedNeXt2D(nn.Module):
    """MedNeXt for 2D slices. Returns logits ``(B, n_classes, H, W)``.

    With ``deep_supervision`` on, a *training-mode* forward returns a list
    ``[full, 1/2, 1/4, 1/8, 1/16]``; eval mode always returns the full-size
    tensor so serving code never has to know about the extra heads.
    """

    def __init__(
        self,
        spec: MedNeXtSpec,
        deep_supervision: bool = False,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.deep_supervision = deep_supervision
        self.grad_checkpoint = grad_checkpoint

        n = spec.n_channels
        k = spec.kernel_size
        r = spec.exp_r
        counts = spec.block_counts
        grn = spec.grn

        total_blocks = sum(counts)
        rates = torch.linspace(0, spec.drop_path_rate, total_blocks).tolist()
        cursor = iter(rates)

        def stage(channels: int, stage_index: int) -> nn.Sequential:
            return nn.Sequential(*[
                MedNeXtBlock(
                    channels, channels, r[stage_index], k,
                    do_res=True, grn=grn, drop_path=next(cursor),
                )
                for _ in range(counts[stage_index])
            ])

        self.stem = nn.Conv2d(spec.in_channels, n, kernel_size=1)

        self.enc_block_0 = stage(n, 0)
        self.down_0 = MedNeXtDownBlock(n, 2 * n, r[1], k, grn=grn)
        self.enc_block_1 = stage(2 * n, 1)
        self.down_1 = MedNeXtDownBlock(2 * n, 4 * n, r[2], k, grn=grn)
        self.enc_block_2 = stage(4 * n, 2)
        self.down_2 = MedNeXtDownBlock(4 * n, 8 * n, r[3], k, grn=grn)
        self.enc_block_3 = stage(8 * n, 3)
        self.down_3 = MedNeXtDownBlock(8 * n, 16 * n, r[4], k, grn=grn)

        self.bottleneck = stage(16 * n, 4)

        self.up_3 = MedNeXtUpBlock(16 * n, 8 * n, r[5], k, grn=grn)
        self.dec_block_3 = stage(8 * n, 5)
        self.up_2 = MedNeXtUpBlock(8 * n, 4 * n, r[6], k, grn=grn)
        self.dec_block_2 = stage(4 * n, 6)
        self.up_1 = MedNeXtUpBlock(4 * n, 2 * n, r[7], k, grn=grn)
        self.dec_block_1 = stage(2 * n, 7)
        self.up_0 = MedNeXtUpBlock(2 * n, n, r[8], k, grn=grn)
        self.dec_block_0 = stage(n, 8)

        self.out_0 = nn.Conv2d(n, spec.n_classes, kernel_size=1)
        if deep_supervision:
            self.out_1 = nn.Conv2d(2 * n, spec.n_classes, kernel_size=1)
            self.out_2 = nn.Conv2d(4 * n, spec.n_classes, kernel_size=1)
            self.out_3 = nn.Conv2d(8 * n, spec.n_classes, kernel_size=1)
            self.out_4 = nn.Conv2d(16 * n, spec.n_classes, kernel_size=1)

    def _run(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            if isinstance(module, nn.Sequential):
                for layer in module:
                    x = checkpoint(layer, x, use_reentrant=False)
                return x
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, list[torch.Tensor]]:
        if x.shape[-1] % DOWNSAMPLE_FACTOR or x.shape[-2] % DOWNSAMPLE_FACTOR:
            raise ValueError(
                f"Input side must be a multiple of {DOWNSAMPLE_FACTOR}; got {tuple(x.shape[-2:])}."
            )
        want_ds = self.deep_supervision and self.training

        x = self.stem(x)
        res_0 = self._run(self.enc_block_0, x)
        x = self._run(self.down_0, res_0)
        res_1 = self._run(self.enc_block_1, x)
        x = self._run(self.down_1, res_1)
        res_2 = self._run(self.enc_block_2, x)
        x = self._run(self.down_2, res_2)
        res_3 = self._run(self.enc_block_3, x)
        x = self._run(self.down_3, res_3)

        x = self._run(self.bottleneck, x)
        ds_4 = self.out_4(x) if want_ds else None

        x = self._run(self.dec_block_3, res_3 + self._run(self.up_3, x))
        ds_3 = self.out_3(x) if want_ds else None
        x = self._run(self.dec_block_2, res_2 + self._run(self.up_2, x))
        ds_2 = self.out_2(x) if want_ds else None
        x = self._run(self.dec_block_1, res_1 + self._run(self.up_1, x))
        ds_1 = self.out_1(x) if want_ds else None
        x = self._run(self.dec_block_0, res_0 + self._run(self.up_0, x))
        out = self.out_0(x)

        if want_ds:
            return [out, ds_1, ds_2, ds_3, ds_4]
        return out


def build_model(
    spec: MedNeXtSpec,
    *,
    deep_supervision: bool = False,
    grad_checkpoint: bool = False,
) -> MedNeXt2D:
    return MedNeXt2D(spec, deep_supervision=deep_supervision, grad_checkpoint=grad_checkpoint)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def strip_deep_supervision(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop the auxiliary heads so the weights load into a serving network."""
    heads = ("out_1.", "out_2.", "out_3.", "out_4.")
    return {k: v for k, v in state_dict.items() if not k.startswith(heads)}


def upkern_transfer(
    target: nn.Module,
    source_state: dict[str, torch.Tensor],
) -> dict[str, list[str]]:
    """Initialise a larger-kernel network from a smaller-kernel checkpoint.

    MedNeXt's UpKern: tensors whose shapes agree are copied; convolution
    kernels that differ only in their spatial extent are bilinearly resampled
    (3x3 -> 5x5). Returns the keys copied, resampled and skipped.
    """
    target_state = target.state_dict()
    copied: list[str] = []
    resampled: list[str] = []
    skipped: list[str] = []

    for key, tensor in target_state.items():
        source = source_state.get(key)
        if source is None:
            skipped.append(key)
            continue
        if source.shape == tensor.shape:
            target_state[key] = source.clone()
            copied.append(key)
        elif source.ndim == 4 and tensor.ndim == 4 and source.shape[:2] == tensor.shape[:2]:
            target_state[key] = F.interpolate(
                source.float(), size=tensor.shape[-2:], mode="bilinear", align_corners=False,
            ).to(tensor.dtype)
            resampled.append(key)
        else:
            skipped.append(key)

    target.load_state_dict(target_state)
    return {"copied": copied, "resampled": resampled, "skipped": skipped}


def spec_summary(spec: MedNeXtSpec, parameters: Optional[int] = None) -> str:
    text = (
        f"MedNeXt-{spec.variant} (2D) | k={spec.kernel_size} | width={spec.n_channels} | "
        f"blocks={list(spec.block_counts)} | exp_r={list(spec.exp_r)} | grn={spec.grn} | "
        f"drop_path={spec.drop_path_rate}"
    )
    if parameters is not None:
        text += f" | {parameters / 1e6:.2f} M params"
    return text
