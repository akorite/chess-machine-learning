"""Training loop: compact records in, losses out, resumable step count."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from ..config import TrainConfig
from ..data.encode import MOVES_LEFT_UNKNOWN, planes_from_bits_batch
from ..distill.prior_kl import prior_kl

LOSS_WEIGHTS = {"policy": 1.0, "promo": 0.2, "wdl": 1.0, "pv": 0.5, "moves_left": 0.1}
PV_PLIES = 6


def batch_from_records(records: Sequence[np.ndarray]) -> dict[str, torch.Tensor]:
    planes = planes_from_bits_batch(
        np.stack([r["pieces"] for r in records]), np.stack([r["meta"] for r in records])
    ).astype(np.float32)
    policy_idx = np.stack([r["policy"] for r in records]).astype(np.int64)
    policy_prob = np.stack([r["policy_prob"] for r in records]).astype(np.float32) / 65535.0
    pv = np.stack([r["pv"] for r in records]).astype(np.int64)
    return {
        "planes": torch.from_numpy(planes),
        "policy_idx": torch.from_numpy(policy_idx),
        "policy_prob": torch.from_numpy(policy_prob),
        "promo": torch.tensor([int(r["promo"]) for r in records], dtype=torch.long),
        "wdl": torch.tensor([int(r["wdl"]) for r in records], dtype=torch.long),
        "pv": torch.from_numpy(pv),
        "moves_left": torch.tensor(
            [int(r["moves_left"]) for r in records], dtype=torch.float32
        ),
    }


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    prior_kl_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    policy_logp = outputs["policy"].log_softmax(-1)
    weights = batch["policy_prob"]
    has_policy = (weights.sum(-1) > 0).float()
    gathered = policy_logp.gather(1, batch["policy_idx"].clamp(0, 4095))
    policy_loss = (-(weights * gathered).sum(-1) * has_policy).sum() / has_policy.sum().clamp(min=1)

    promo_mask = (batch["promo"] < 4).float()
    promo_loss = (
        nn.functional.cross_entropy(outputs["promo"], batch["promo"].clamp(0, 3), reduction="none")
        * promo_mask
    ).sum() / promo_mask.sum().clamp(min=1)

    wdl_mask = (batch["wdl"] < 3).float()
    wdl_loss = (
        nn.functional.cross_entropy(outputs["wdl"], batch["wdl"].clamp(0, 2), reduction="none")
        * wdl_mask
    ).sum() / wdl_mask.sum().clamp(min=1)

    pv_loss = outputs["policy"].new_zeros(())
    pv_count = outputs["policy"].new_zeros(())
    for ply in range(min(PV_PLIES, outputs["pv"].shape[1])):
        target = batch["pv"][:, ply]
        valid = (target > 0).float()
        pv_loss = pv_loss + (
            nn.functional.cross_entropy(outputs["pv"][:, ply], target.clamp(0, 4095), reduction="none")
            * valid
        ).sum()
        pv_count = pv_count + valid.sum()
    pv_loss = pv_loss / pv_count.clamp(min=1)

    ml_mask = (batch["moves_left"] < MOVES_LEFT_UNKNOWN).float()
    ml_target = batch["moves_left"].clamp(max=MOVES_LEFT_UNKNOWN - 1) / 100.0
    ml_loss = ((outputs["moves_left"] - ml_target) ** 2 * ml_mask).sum() / ml_mask.sum().clamp(min=1)

    # Forward KL to the teacher prior over the recorded top-8 support (KTD11),
    # computed on labeled rows only — unlabeled rows have no meaningful prior.
    labeled = has_policy.bool()
    if labeled.any():
        support = torch.zeros_like(outputs["policy"], dtype=torch.bool)
        support.scatter_(1, batch["policy_idx"], batch["policy_prob"] > 0)
        teacher_logits = torch.zeros_like(outputs["policy"], dtype=torch.float32)
        teacher_logits.scatter_(
            1, batch["policy_idx"], batch["policy_prob"].float().clamp(min=1e-9).log()
        )
        kl = prior_kl(teacher_logits[labeled], outputs["policy"][labeled].float(), support[labeled])
    else:
        kl = outputs["policy"].new_zeros(())

    total = (
        LOSS_WEIGHTS["policy"] * policy_loss
        + LOSS_WEIGHTS["promo"] * promo_loss
        + LOSS_WEIGHTS["wdl"] * wdl_loss
        + LOSS_WEIGHTS["pv"] * pv_loss
        + LOSS_WEIGHTS["moves_left"] * ml_loss
        + prior_kl_weight * kl
    )
    return total, {
        "loss": float(total.detach()),
        "policy": float(policy_loss.detach()),
        "promo": float(promo_loss.detach()),
        "wdl": float(wdl_loss.detach()),
        "pv": float(pv_loss.detach()),
        "moves_left": float(ml_loss.detach()),
        "prior_kl": float(kl.detach()),
    }


def learning_rate(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    decay_steps = max(cfg.max_steps - cfg.warmup_steps, 1)
    progress = min((step - cfg.warmup_steps) / decay_steps, 1.0)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def batch_slice(dataset: Sequence[np.ndarray], step: int, batch_size: int) -> list[np.ndarray]:
    start = (step * batch_size) % len(dataset)
    indices = [(start + i) % len(dataset) for i in range(batch_size)]
    return [dataset[i] for i in indices]


def train(
    model: nn.Module,
    dataset: Sequence[np.ndarray],
    cfg: TrainConfig,
    steps: int,
    device: str = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
    start_step: int = 0,
) -> dict[str, float]:
    model.to(device)
    model.train()
    train_model = torch.compile(model) if cfg.compile and torch.device(device).type == "cuda" else model
    optimizer = optimizer or torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    use_bf16 = torch.device(device).type == "cuda"
    metrics: dict[str, float] = {}
    started = time.time()
    for offset in range(steps):
        step = start_step + offset
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, cfg)
        batch = {k: v.to(device) for k, v in batch_from_records(batch_slice(dataset, step, cfg.batch_size)).items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            outputs = train_model(batch["planes"])
        loss, metrics = compute_loss(outputs, batch, prior_kl_weight=cfg.prior_kl_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
    metrics["steps"] = float(steps)
    metrics["positions_per_second"] = steps * cfg.batch_size / max(time.time() - started, 1e-9)
    return metrics
