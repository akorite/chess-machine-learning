"""Output heads: policy, promotion, result, value, PV plies, moves left."""

from __future__ import annotations

import torch
from torch import nn


class Heads(nn.Module):
    """PV plies use a rank-4 bilinear form on projections separate from the policy ones."""

    def __init__(self, dim: int, pv_plies: int = 6, pv_rank: int = 4):
        super().__init__()
        self.src = nn.Linear(dim, dim, bias=False)
        self.dst = nn.Linear(dim, dim, bias=False)
        self.promo = nn.Linear(dim, 4)
        self.wdl = nn.Linear(dim, 3)
        self.value = nn.Linear(dim, 1)
        self.moves_left = nn.Linear(dim, 1)
        self.pv_src = nn.Linear(dim, pv_rank, bias=False)
        self.pv_dst = nn.Linear(dim, pv_rank, bias=False)
        self.pv_left = nn.Parameter(torch.empty(pv_plies, pv_rank, pv_rank).normal_(std=0.02))
        self.pv_right = nn.Parameter(torch.empty(pv_plies, pv_rank, pv_rank).normal_(std=0.02))

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        src = self.src(tokens)
        dst = self.dst(tokens)
        policy = torch.einsum("bic,bjc->bij", src, dst).flatten(1)
        left = torch.einsum("bfr,prs->bpfs", self.pv_src(tokens), self.pv_left)
        right = torch.einsum("bgr,prs->bpgs", self.pv_dst(tokens), self.pv_right)
        pv = torch.einsum("bpfs,bpgs->bpfg", left, right).flatten(2)
        pooled = tokens.mean(dim=1)
        return {
            "policy": policy,
            "promo": self.promo(pooled),
            "wdl": self.wdl(pooled),
            "value": torch.tanh(self.value(pooled)).squeeze(-1),
            "pv": pv,
            "moves_left": self.moves_left(pooled).squeeze(-1),
        }
