"""Output heads: policy, promotion, result, PV plies, moves left."""

from __future__ import annotations

import torch
from torch import nn


class Heads(nn.Module):
    """PV plies use a rank-4 bilinear form; the full dim x dim form costs ~200x more MACs."""

    def __init__(self, dim: int, pv_plies: int = 6, pv_rank: int = 4):
        super().__init__()
        self.src = nn.Linear(dim, dim, bias=False)
        self.dst = nn.Linear(dim, dim, bias=False)
        self.promo = nn.Linear(dim, 4)
        self.wdl = nn.Linear(dim, 3)
        self.moves_left = nn.Linear(dim, 1)
        self.pv_left = nn.Parameter(torch.zeros(pv_plies, dim, pv_rank))
        self.pv_right = nn.Parameter(torch.zeros(pv_plies, dim, pv_rank))

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        src = self.src(tokens)
        dst = self.dst(tokens)
        policy = torch.einsum("bic,bjc->bij", src, dst).flatten(1)
        left = torch.einsum("bfi,pir->bpfr", src, self.pv_left)
        right = torch.einsum("bgj,pjr->bpgr", dst, self.pv_right)
        pv = torch.einsum("bpfr,bpgr->bpfg", left, right).flatten(2)
        pooled = tokens.mean(dim=1)
        return {
            "policy": policy,
            "promo": self.promo(pooled),
            "wdl": self.wdl(pooled),
            "pv": pv,
            "moves_left": self.moves_left(pooled).squeeze(-1),
        }
