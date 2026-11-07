"""Compare two graphs on held-out positions: move agreement, value MAE, latency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from ..train.loop import batch_from_records
from ..train.throughput import Throughput, measure


@dataclass(frozen=True)
class QuantReport:
    move_agreement: float
    value_mae_delta: float
    latency_fp_ms: float
    latency_int8_ms: float


def value_mae(outputs: dict[str, torch.Tensor], wdl: torch.Tensor) -> float:
    expected = (outputs["wdl"].softmax(-1) * torch.tensor([-1.0, 0.0, 1.0])).sum(-1)
    target = wdl.float() - 1.0
    known = wdl < 3
    if not known.any():
        return float("nan")
    return float((expected[known] - target[known]).abs().mean())


def compare(
    reference: nn.Module,
    candidate: nn.Module,
    dataset: Sequence[np.ndarray],
    limit: int = 512,
    batch_size: int = 64,
) -> QuantReport:
    reference.eval()
    candidate.eval()
    agree = total = 0
    mae_reference: list[float] = []
    mae_candidate: list[float] = []
    with torch.no_grad():
        for start in range(0, min(len(dataset), limit), batch_size):
            records = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            batch = batch_from_records(records)
            ref_out = reference(batch["planes"])
            cand_out = candidate(batch["planes"])
            agree += int((ref_out["policy"].argmax(-1) == cand_out["policy"].argmax(-1)).sum())
            total += batch["planes"].shape[0]
            ref_mae = value_mae(ref_out, batch["wdl"])
            cand_mae = value_mae(cand_out, batch["wdl"])
            if not np.isnan(ref_mae):
                mae_reference.append(ref_mae)
            if not np.isnan(cand_mae):
                mae_candidate.append(cand_mae)
    fp: Throughput = measure(reference)
    int8: Throughput = measure(candidate)
    return QuantReport(
        move_agreement=agree / max(total, 1),
        value_mae_delta=abs(float(np.mean(mae_candidate)) - float(np.mean(mae_reference)))
        if mae_reference
        else float("nan"),
        latency_fp_ms=fp.latency_ms_batch1,
        latency_int8_ms=int8.latency_ms_batch1,
    )
