"""Output heads: policy, promotion, result, PV plies, moves left."""

from __future__ import annotations

import torch
from torch import nn


class Heads(nn.Module):
    def __init__(self, dim: int, pv_plies: int = 6):
        super().__init__()
        self.src = nn.Linear(dim, dim, bias=False)
        self.dst = nn.Linear(dim, dim, bias=False)
        self.promo = nn.Linear(dim, 4)
        self.wdl = nn.Linear(dim, 3)
        self.moves_left = nn.Linear(dim, 1)
        self.pv_weights = nn.Parameter(torch.zeros(pv_plies, dim, dim))

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        src = self.src(tokens)
        dst = self.dst(tokens)
        policy = torch.einsum("bic,bjc->bij", src, dst).flatten(1)
        pv = torch.einsum("bfi,pij,bgj->bpfg", src, self.pv_weights, dst).flatten(2)
        pooled = tokens.mean(dim=1)
        return {
            "policy": policy,
            "promo": self.promo(pooled),
            "wdl": self.wdl(pooled),
            "pv": pv,
            "moves_left": self.moves_left(pooled).squeeze(-1),
        }
