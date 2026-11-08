"""Hybrid ResNet+transformer student, plus a pure-transformer ablation."""

from __future__ import annotations

import torch
from torch import nn

from ..config import Preset
from ..data.encode import PLANES
from .heads import Heads
from .smolgen import Block
from .stem import Stem


class Student(nn.Module):
    def __init__(self, preset: Preset, pure_transformer: bool = False):
        super().__init__()
        self.preset = preset
        self.stem = None if pure_transformer else Stem(preset.stem_channels, preset.stem_blocks)
        in_channels = PLANES if pure_transformer else preset.stem_channels
        self.token_proj = nn.Conv2d(in_channels, preset.d_model, 1)
        self.pos_embed = nn.Parameter(
            torch.empty(1, 64, preset.d_model).normal_(std=0.02)  # 8x8 squares
        )
        self.blocks = nn.ModuleList(
            Block(preset.d_model, preset.n_heads) for _ in range(preset.n_layers)
        )
        self.heads = Heads(preset.d_model, preset.pv_plies)

    def embed(self, planes: torch.Tensor) -> torch.Tensor:
        """Board planes to (batch, 64, d_model) square tokens carrying their square identity."""
        x = planes.float()
        if self.stem is not None:
            x = self.stem(x)
        return self.token_proj(x).flatten(2).transpose(1, 2) + self.pos_embed

    def encode(self, planes: torch.Tensor) -> torch.Tensor:
        tokens = self.embed(planes)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens

    def forward(self, planes: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.heads(self.encode(planes))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
