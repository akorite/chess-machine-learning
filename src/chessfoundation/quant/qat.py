"""Quantisation-aware training with per-tensor fake quantisation."""

from __future__ import annotations

import torch
from torch import nn


class _FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        scale = x.abs().max() / 127.0
        if scale == 0:
            return x
        return torch.round(x / scale).clamp(-127, 127) * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def fake_quant(x: torch.Tensor) -> torch.Tensor:
    return _FakeQuant.apply(x)


class QATLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(fake_quant(x), fake_quant(self.linear.weight), self.linear.bias)


class QATConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv2d(
            fake_quant(x), fake_quant(self.conv.weight), self.conv.bias, self.conv.stride,
            self.conv.padding, self.conv.dilation, self.conv.groups,
        )


def prepare_qat(model: nn.Module) -> nn.Module:
    """Replace conv and linear layers in place; heads stay float at inference."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(model, name, QATConv2d(child))
        elif isinstance(child, nn.Linear):
            setattr(model, name, QATLinear(child))
        else:
            prepare_qat(child)
    return model


def convert(model: nn.Module) -> nn.Module:
    """Unwrap fake quantisation; weights keep the QAT-trained values."""
    for name, child in list(model.named_children()):
        if isinstance(child, QATConv2d):
            setattr(model, name, child.conv)
        elif isinstance(child, QATLinear):
            setattr(model, name, child.linear)
        else:
            convert(child)
    return model


def is_quantised(model: nn.Module) -> bool:
    return any(isinstance(m, (QATLinear, QATConv2d)) for m in model.modules())
