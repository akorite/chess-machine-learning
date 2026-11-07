"""Smolgen attention: board-wide bias learned from the token set."""

from __future__ import annotations

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class SmolgenAttention(nn.Module):
    def __init__(self, dim: int, heads: int, gen_size: int = 32, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.gen_size = gen_size
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.gen = nn.Linear(dim, heads * gen_size)
        self.gen_left = nn.Linear(gen_size, 64)
        self.gen_right = nn.Linear(gen_size, 64)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        head_dim = dim // self.heads
        q = q.reshape(batch, tokens, self.heads, head_dim).transpose(1, 2)
        k = k.reshape(batch, tokens, self.heads, head_dim).transpose(1, 2)
        v = v.reshape(batch, tokens, self.heads, head_dim).transpose(1, 2)

        gen = self.gen(x.mean(dim=1)).reshape(batch, self.heads, self.gen_size)
        left = self.gen_left(gen)
        right = self.gen_right(gen)
        bias = torch.einsum("bhl,bhr->bhlr", left, right) / math.sqrt(self.gen_size)

        attended = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0 / math.sqrt(head_dim),
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, dim)
        return self.out(attended)


class Block(nn.Module):
    """Pre-norm transformer block, FFN at 1x width, no linear bias."""

    def __init__(self, dim: int, heads: int, gen_size: int = 32, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SmolgenAttention(dim, heads, gen_size, dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))
