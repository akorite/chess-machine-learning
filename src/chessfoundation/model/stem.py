"""Residual stem: board planes to square tokens."""

from __future__ import annotations

import torch
from torch import nn

from ..data.encode import PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return torch.relu(x + residual)


class Stem(nn.Module):
    """Spatial inductive bias before square-token attention."""

    def __init__(self, channels: int = 64, blocks: int = 6):
        super().__init__()
        self.input = nn.Conv2d(PLANES, channels, 3, padding=1)
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.input(planes))
