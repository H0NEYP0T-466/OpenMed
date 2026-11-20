"""MiT-B0 / SegFormer-B0 encoder, fused 1/4-resolution decoder, two sigmoid heads.

Architecture
    encoder    HuggingFace ``SegformerModel`` (Mix Transformer B0, ImageNet weights
               when training). Four feature maps at strides 4, 8, 16, 32.
    decoder    SegFormer all-MLP decoder (1x1 projection per stage, bilinear
               upsample to 1/4, concat, 1x1 fuse) with GroupNorm instead of
               BatchNorm. Head B sees ~13 % of the pool and serving runs batch=1;
               batch statistics would couple the two heads and the two inputs.
    heads      ``lung`` (every image, full frame) and ``opacity`` (supervised
               subset, lung-bbox crop). Independent sigmoids, never a softmax.

One forward call runs both passes (``forward(full, crop)``): the encoder sees the
concatenated batch once, the lung head reads the full-frame rows and the opacity
head reads the crop rows. That keeps a single autograd graph per step, which is
what DDP needs, and it makes the two-pass cost explicit instead of hidden.

Checkpoint portability
    The ``transformers`` parameter names changed between releases
    (``encoder.block.N.M.layer_norm_1`` became ``stages.N.blocks.M.layernorm_before``
    and so on). A checkpoint trained on Kaggle must load on a different install,
    so weights are stored under one canonical naming (the long-standing
    ``encoder.*`` layout) and translated to whichever layout is installed. The
    translation is exact or it raises; nothing is skipped.

``transformers`` is imported lazily, only when a network is built.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels import HEADS

ENCODER_MIT_B0 = "mit-b0"
ENCODER_DEBUG = "debug-tiny"
ENCODER_KINDS = (ENCODER_MIT_B0, ENCODER_DEBUG)
DEFAULT_PRETRAINED_SOURCE = "nvidia/mit-b0"
MODEL_TAG = "segformer-mit-b0-two-head"


@dataclass(frozen=True)
class SegSpec:
    """The exact architecture, serialised into every checkpoint."""

    encoder: str = ENCODER_MIT_B0
    heads: tuple[str, ...] = HEADS
    hidden_sizes: tuple[int, ...] = (32, 64, 160, 256)
    depths: tuple[int, ...] = (2, 2, 2, 2)
    num_attention_heads: tuple[int, ...] = (1, 2, 5, 8)
    sr_ratios: tuple[int, ...] = (8, 4, 2, 1)
    patch_sizes: tuple[int, ...] = (7, 3, 3, 3)
    strides: tuple[int, ...] = (4, 2, 2, 2)
    mlp_ratios: tuple[int, ...] = (4, 4, 4, 4)
    decoder_dim: int = 256
    head_dim: int = 64
    # Regularisation. Rates are training-time only; they do not change parameter
    # shapes, so serving can rebuild with the same values and call ``.eval()``.
    drop_path: float = 0.1
    decoder_dropout: float = 0.1
    head_dropout: float = 0.1
    encoder_dropout: float = 0.0
    attention_dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SegSpec:
        tuples = (
            "heads", "hidden_sizes", "depths", "num_attention_heads", "sr_ratios",
            "patch_sizes", "strides", "mlp_ratios",
        )
        data = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        for key in tuples:
            if key in data:
                data[key] = tuple(data[key])
        spec = cls(**data)
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.encoder not in ENCODER_KINDS:
            raise ValueError(f"Unknown encoder {self.encoder!r}; expected one of {ENCODER_KINDS}.")
        if not self.heads or set(self.heads) - set(HEADS) or self.heads[0] != "lung":
            raise ValueError(
                f"heads must be ('lung',) or ('lung', 'opacity'), got {self.heads!r}."
            )
        if len(set(self.heads)) != len(self.heads):
            raise ValueError(f"duplicate heads in {self.heads!r}.")
        stages = len(self.hidden_sizes)
        for name in ("depths", "num_attention_heads", "sr_ratios", "patch_sizes", "strides",
                     "mlp_ratios"):
            if len(getattr(self, name)) != stages:
                raise ValueError(f"{name} must have {stages} entries.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------ key translation

# Canonical (long-standing ``encoder.*``) -> newer ``stages.*`` layout.
_CANONICAL_TO_STAGES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        (r"^encoder\.patch_embeddings\.(\d+)\.(proj|layer_norm)\.", r"stages.\1.patch_embeddings.\2."),
        (r"^encoder\.block\.(\d+)\.(\d+)\.layer_norm_1\.", r"stages.\1.blocks.\2.layernorm_before."),
        (r"^encoder\.block\.(\d+)\.(\d+)\.layer_norm_2\.", r"stages.\1.blocks.\2.layernorm_after."),
        (r"^encoder\.block\.(\d+)\.(\d+)\.attention\.self\.query\.", r"stages.\1.blocks.\2.attention.q_proj."),
        (r"^encoder\.block\.(\d+)\.(\d+)\.attention\.self\.key\.", r"stages.\1.blocks.\2.attention.k_proj."),
        (r"^encoder\.block\.(\d+)\.(\d+)\.attention\.self\.value\.", r"stages.\1.blocks.\2.attention.v_proj."),
        (r"^encoder\.block\.(\d+)\.(\d+)\.attention\.self\.sr\.",
         r"stages.\1.blocks.\2.attention.sequence_reduction.sequence_reduction."),
        (r"^encoder\.block\.(\d+)\.(\d+)\.attention\.self\.layer_norm\.",
         r"stages.\1.blocks.\2.attention.sequence_reduction.layer_norm."),
        (r"^encoder\.block\.(\d+)\.(\d+)\.attention\.output\.dense\.", r"stages.\1.blocks.\2.attention.o_proj."),
        (r"^encoder\.block\.(\d+)\.(\d+)\.mlp\.dense1\.", r"stages.\1.blocks.\2.mlp.fc1."),
        (r"^encoder\.block\.(\d+)\.(\d+)\.mlp\.dense2\.", r"stages.\1.blocks.\2.mlp.fc2."),
        (r"^encoder\.block\.(\d+)\.(\d+)\.mlp\.dwconv\.dwconv\.", r"stages.\1.blocks.\2.mlp.dwconv.dwconv."),
        (r"^encoder\.layer_norm\.(\d+)\.", r"stages.\1.layer_norm."),
    )
)
def _apply(rules: Sequence[tuple[re.Pattern[str], str]], key: str) -> str | None:
    for pattern, replacement in rules:
        if pattern.match(key):
            return pattern.sub(replacement, key, count=1)
    return None


def canonical_to_stages(key: str) -> str | None:
    return _apply(_CANONICAL_TO_STAGES, key)


_STAGES_TO_CANONICAL: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        (r"^stages\.(\d+)\.patch_embeddings\.(proj|layer_norm)\.", r"encoder.patch_embeddings.\1.\2."),
        (r"^stages\.(\d+)\.blocks\.(\d+)\.layernorm_before\.", r"encoder.block.\1.\2.layer_norm_1."),
        (r"^stages\.(\d+)\.blocks\.(\d+)\.layernorm_after\.", r"encoder.block.\1.\2.layer_norm_2."),
        (r"^stages\.(\d+)\.blocks\.(\d+)\.attention\.q_proj\.", r"encoder.block.\1.\2.attention.self.query."),
        (r"^stages\.(\d+)\.blocks\.(\d+)\.attention\.k_proj\.", r"encoder.block.\1.\2.attention.self.key."),
        (r"^stages\.(\d+)\.blocks\.(\d+)\.attention\.v_proj\.", r"encoder.block.\1.\2.attention.self.value."),
        (r"^stages\.(\d+)\.blocks\.(\d+)\.attention\.sequence_reduction\.sequence_reduction\.",
         r"encoder.block.\1.\2.attention.self.sr."),
        (r"^stages\.(\d+)\.blocks\.(\d+)\.attention\.sequence_reduction\.layer_norm\.",
         r"encoder.block.\1.\2.attention.self.layer_norm."),
        (r"^stages\.(\d+)\.blocks\.(\d+)\.attention\.o_proj\.", r"encoder.block.\1.\2.attention.output.dense."),
        (r"^stages\.(\d+)\.blocks\.(\d+)\.mlp\.fc1\.", r"encoder.block.\1.\2.mlp.dense1."),
        (r"^stages\.(\d+)\.blocks\.(\d+)\.mlp\.fc2\.", r"encoder.block.\1.\2.mlp.dense2."),
        (r"^stages\.(\d+)\.blocks\.(\d+)\.mlp\.dwconv\.dwconv\.", r"encoder.block.\1.\2.mlp.dwconv.dwconv."),
        (r"^stages\.(\d+)\.layer_norm\.", r"encoder.layer_norm.\1."),
    )
)


def stages_to_canonical(key: str) -> str | None:
    return _apply(_STAGES_TO_CANONICAL, key)


def detect_hf_layout(keys: Sequence[str]) -> str:
    """``"legacy"`` (``encoder.*``) or ``"stages"`` (``stages.*``); raises otherwise."""
    if any(k.startswith("encoder.block.") for k in keys):
        return "legacy"
    if any(k.startswith("stages.") for k in keys):
        return "stages"
    sample = ", ".join(list(keys)[:4])
    raise RuntimeError(
        "Installed `transformers` exposes a SegformerModel parameter layout this build does not "
        f"know (first keys: {sample}). Pin transformers to a release with a known layout."
    )


def to_canonical(state: dict[str, torch.Tensor], layout: str) -> dict[str, torch.Tensor]:
    if layout == "legacy":
        return dict(state)
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        mapped = stages_to_canonical(key)
        if mapped is None:
            raise RuntimeError(f"No canonical name for installed encoder parameter {key!r}.")
        out[mapped] = value
    return out


def from_canonical(state: dict[str, torch.Tensor], layout: str) -> dict[str, torch.Tensor]:
    if layout == "legacy":
        return dict(state)
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        mapped = canonical_to_stages(key)
        if mapped is None:
            raise RuntimeError(f"No installed-layout name for canonical encoder parameter {key!r}.")
        out[mapped] = value
    return out


def _groups(channels: int) -> int:
    """Largest GroupNorm group count (<= 32) that divides ``channels``."""
    for candidate in (32, 16, 8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


# ----------------------------------------------------------------------- encoders


class HFMiTEncoder(nn.Module):
    """Thin wrapper over ``transformers.SegformerModel`` returning four feature maps."""

    def __init__(self, spec: SegSpec, pretrained_source: str | None = None) -> None:
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerModel
        except ImportError as exc:  # pragma: no cover - exercised only without transformers
            raise RuntimeError(
                "The MiT-B0 encoder needs the `transformers` package (it is listed in "
                "backend/requirements.txt). Install it, or build the debug encoder in tests."
            ) from exc

        overrides = {
            "drop_path_rate": spec.drop_path,
            "hidden_dropout_prob": spec.encoder_dropout,
            "attention_probs_dropout_prob": spec.attention_dropout,
        }
        if pretrained_source:
            self.hf = SegformerModel.from_pretrained(pretrained_source, **overrides)
            self.pretrained_source: str | None = pretrained_source
            expected = (tuple(self.hf.config.hidden_sizes), tuple(self.hf.config.depths))
            if expected != (spec.hidden_sizes, spec.depths):
                raise RuntimeError(
                    f"Pretrained encoder {pretrained_source!r} has hidden_sizes/depths {expected}, "
                    f"which is not the MiT-B0 layout {(spec.hidden_sizes, spec.depths)}."
                )
        else:
            config = SegformerConfig(
                num_channels=3,
                num_encoder_blocks=len(spec.hidden_sizes),
                depths=list(spec.depths),
                sr_ratios=list(spec.sr_ratios),
                hidden_sizes=list(spec.hidden_sizes),
                patch_sizes=list(spec.patch_sizes),
                strides=list(spec.strides),
                num_attention_heads=list(spec.num_attention_heads),
                mlp_ratios=list(spec.mlp_ratios),
                reshape_last_stage=True,
                **overrides,
            )
            self.hf = SegformerModel(config)
            self.pretrained_source = None
        self.layout = detect_hf_layout(list(self.hf.state_dict().keys()))
        self.out_channels = tuple(spec.hidden_sizes)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        h = x
        if self.layout == "stages":
            for stage in self.hf.stages:
                h = stage(h)
                features.append(h)
            return features
        enc = self.hf.encoder
        batch = x.shape[0]
        for embed, blocks, norm in zip(enc.patch_embeddings, enc.block, enc.layer_norm, strict=True):
            h, height, width = embed(h)
            for block in blocks:
                h = block(h, height, width)[0]
            h = norm(h)
            h = h.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
            features.append(h)
        return features

    def canonical_state_dict(self) -> dict[str, torch.Tensor]:
        return to_canonical(self.hf.state_dict(), self.layout)

    def load_canonical(self, state: dict[str, torch.Tensor]) -> None:
        self.hf.load_state_dict(from_canonical(state, self.layout), strict=True)


class DebugEncoder(nn.Module):
    """Tiny convolutional pyramid with the encoder interface.

    Exists so the whole pipeline (hygiene, split, cache, train, tune, test, plots,
    ZIP, serving) can be smoke-tested on CPU in seconds. It is never a result:
    serving refuses a checkpoint built on it unless explicitly allowed.
    """

    def __init__(self, spec: SegSpec) -> None:
        super().__init__()
        chans = spec.hidden_sizes
        layers: list[nn.Module] = []
        previous = 3
        for width, stride in zip(chans, spec.strides, strict=True):
            layers.append(nn.Sequential(
                nn.Conv2d(previous, width, kernel_size=3, stride=stride, padding=1),
                nn.GroupNorm(_groups(width), width),
                nn.GELU(),
            ))
            previous = width
        self.stages = nn.ModuleList(layers)
        self.out_channels = tuple(chans)
        self.layout = "debug"
        self.pretrained_source = None

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

    def canonical_state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self.state_dict())

    def load_canonical(self, state: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state, strict=True)


# ------------------------------------------------------------------------ network


class FusedDecoder(nn.Module):
    """SegFormer all-MLP decoder fused at 1/4 resolution."""

    def __init__(self, in_channels: Sequence[int], dim: int, dropout: float) -> None:
        super().__init__()
        self.project = nn.ModuleList(nn.Conv2d(c, dim, kernel_size=1) for c in in_channels)
        self.fuse = nn.Conv2d(dim * len(in_channels), dim, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_groups(dim), dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        target = features[0].shape[-2:]
        parts = []
        for feature, proj in zip(features, self.project, strict=True):
            y = proj(feature)
            if y.shape[-2:] != target:
                y = F.interpolate(y, size=target, mode="bilinear", align_corners=False)
            parts.append(y)
        fused = self.fuse(torch.cat(parts[::-1], dim=1))
        return self.dropout(self.act(self.norm(fused)))


class SegHead(nn.Module):
    """3x3 refine, then a 1x1 logit. One per task; the heads share nothing but the decoder."""

    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.refine = nn.Conv2d(in_dim, hidden, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(_groups(hidden), hidden)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.logit = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logit(self.dropout(self.act(self.norm(self.refine(x)))))


class LungSegNet(nn.Module):
    """Shared encoder and decoder, one independent sigmoid head per task."""

    def __init__(self, spec: SegSpec, pretrained_source: str | None = None) -> None:
        super().__init__()
        spec.validate()
        self.spec = spec
        if spec.encoder == ENCODER_MIT_B0:
            self.encoder: HFMiTEncoder | DebugEncoder = HFMiTEncoder(spec, pretrained_source)
        else:
            if pretrained_source:
                raise ValueError("The debug encoder has no pretrained weights.")
            self.encoder = DebugEncoder(spec)
        self.decoder = FusedDecoder(self.encoder.out_channels, spec.decoder_dim, spec.decoder_dropout)
        self.head_lung = SegHead(spec.decoder_dim, spec.head_dim, spec.head_dropout)
        self.head_opacity = (
            SegHead(spec.decoder_dim, spec.head_dim, spec.head_dropout)
            if "opacity" in spec.heads else None
        )

    @property
    def has_opacity(self) -> bool:
        return self.head_opacity is not None

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    @staticmethod
    def _upsample(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)

    def forward(
        self, full: torch.Tensor, crop: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(lung_logits, opacity_logits)``, each ``(B, H, W)`` at input resolution.

        ``full`` feeds the lung head. ``crop`` (lung-bbox crops resized to the input
        size) feeds the opacity head; when absent the second pass is skipped and
        ``None`` is returned in its place.
        """
        size = (full.shape[-2], full.shape[-1])
        if crop is None or self.head_opacity is None:
            features = self._decode(full)
            return self._upsample(self.head_lung(features), size)[:, 0], None
        both = torch.cat([full, crop], dim=0)
        features = self._decode(both)
        n_full = full.shape[0]
        lung = self._upsample(self.head_lung(features[:n_full]), size)[:, 0]
        opacity = self._upsample(self.head_opacity(features[n_full:]), size)[:, 0]
        return lung, opacity

    # -- single-head convenience for serving (each pass is independent) -------------

    @torch.no_grad()
    def lung_logits(self, full: torch.Tensor) -> torch.Tensor:
        return self.forward(full, None)[0]

    @torch.no_grad()
    def opacity_logits(self, crop: torch.Tensor) -> torch.Tensor:
        if self.head_opacity is None:
            raise RuntimeError("This network has no opacity head.")
        size = (crop.shape[-2], crop.shape[-1])
        return self._upsample(self.head_opacity(self._decode(crop)), size)[:, 0]

    # -- canonical checkpoint (de)serialisation --------------------------------------

    def canonical_state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for key, value in self.encoder.canonical_state_dict().items():
            state[f"encoder.{key}"] = value
        for name in ("decoder", "head_lung", "head_opacity"):
            module = getattr(self, name)
            if module is None:
                continue
            for key, value in module.state_dict().items():
                state[f"{name}.{key}"] = value
        return state

    def load_canonical(self, state: dict[str, torch.Tensor]) -> None:
        """Strict load: every key present, every key consumed, shapes equal."""
        encoder_state: dict[str, torch.Tensor] = {}
        rest: dict[str, dict[str, torch.Tensor]] = {}
        for key, value in state.items():
            head, _, tail = key.partition(".")
            if head == "encoder":
                encoder_state[tail] = value
            else:
                rest.setdefault(head, {})[tail] = value
        known = {"decoder", "head_lung"} | ({"head_opacity"} if self.head_opacity is not None else set())
        unknown = set(rest) - known
        if unknown:
            raise RuntimeError(
                f"Checkpoint carries weights for {sorted(unknown)} but the declared architecture "
                f"has heads {list(self.spec.heads)}."
            )
        missing = known - set(rest)
        if missing:
            raise RuntimeError(f"Checkpoint is missing weights for {sorted(missing)}.")
        self.encoder.load_canonical(encoder_state)
        for name in sorted(known):
            getattr(self, name).load_state_dict(rest[name], strict=True)


def build_model(spec: SegSpec, *, pretrained_source: str | None = None) -> LungSegNet:
    return LungSegNet(spec, pretrained_source=pretrained_source)


def build_debug_spec(heads: tuple[str, ...] = HEADS) -> SegSpec:
    """Smallest legal spec for tests and CPU smoke runs. Never a result."""
    return SegSpec(
        encoder=ENCODER_DEBUG, heads=heads, hidden_sizes=(8, 12, 16, 24), decoder_dim=16,
        head_dim=8, drop_path=0.0, decoder_dropout=0.0, head_dropout=0.0,
    )

