"""Stage gates: metric improvement, stage source discipline, budget projection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..budget import BudgetState, check_run
from ..config import BudgetConfig
from .loop import batch_from_records


@dataclass(frozen=True)
class StageMetrics:
    policy_accuracy: float
    pv_accuracy: float
    value_mae: float


def evaluate(
    model: nn.Module, dataset: Sequence[np.ndarray], limit: int = 512, batch_size: int = 64
) -> StageMetrics:
    model.eval()
    policy_hits = policy_total = pv_hits = pv_total = 0
    value_errors: list[float] = []
    with torch.no_grad():
        for start in range(0, min(len(dataset), limit), batch_size):
            records = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            batch = batch_from_records(records)
            outputs = model(batch["planes"])
            policy_arg = outputs["policy"].argmax(-1)
            has_policy = batch["policy_prob"].sum(-1) > 0
            policy_hits += int((policy_arg == batch["policy_idx"][:, 0])[has_policy].sum())
            policy_total += int(has_policy.sum())
            pv_arg = outputs["pv"][:, 0].argmax(-1)
            pv_valid = batch["pv"][:, 0] > 0
            pv_hits += int((pv_arg == batch["pv"][:, 0])[pv_valid].sum())
            pv_total += int(pv_valid.sum())
            expected = (outputs["wdl"].softmax(-1) * torch.tensor([-1.0, 0.0, 1.0])).sum(-1)
            target = batch["wdl"].float() - 1.0
            known = batch["wdl"] < 3
            value_errors.extend((expected[known] - target[known]).abs().tolist())
    return StageMetrics(
        policy_accuracy=policy_hits / max(policy_total, 1),
        pv_accuracy=pv_hits / max(pv_total, 1),
        value_mae=float(np.mean(value_errors)) if value_errors else float("nan"),
    )


class GateFailed(RuntimeError):
    pass


def gate(previous: StageMetrics | None, current: StageMetrics, minimum_policy_gain: float = 0.0) -> None:
    if previous is None:
        return
    if current.policy_accuracy + minimum_policy_gain < previous.policy_accuracy:
        raise GateFailed(
            f"policy accuracy regressed {previous.policy_accuracy:.4f} -> {current.policy_accuracy:.4f}"
        )
    if current.value_mae > previous.value_mae:
        raise GateFailed(f"value MAE regressed {previous.value_mae:.4f} -> {current.value_mae:.4f}")


def check_stage_source(stage: str, counts: dict[str, int]) -> None:
    if stage == "deep":
        if counts.get("24plus", 0) == 0 and counts.get("knodes100k", 0) == 0:
            raise GateFailed("deep stage has no depth>=24 or knodes>=100k rows")
        if counts.get("lt20", 0) > 0:
            raise GateFailed("deep stage source contains sub-depth-20 rows")


def project_stage(
    cfg: BudgetConfig, state: BudgetState, accel: str, hours: float, phase: str = "pretrain"
) -> float:
    return check_run(cfg, state, accel, hours, phase)
