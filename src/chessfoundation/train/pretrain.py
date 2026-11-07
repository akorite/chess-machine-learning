"""Stage A self-supervised masking, then value and deep-label stages."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from ..config import RunConfig, TrainConfig
from ..data.cache import ShardDataset
from ..model.presets import build_student
from . import loop
from .checkpoint import save
from .gates import check_stage_source

STAGES = ("ssl", "value", "deep")


class MaskedPieceDecoder(nn.Module):
    """Predict piece planes from masked tokens."""

    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Linear(dim, 12)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(tokens)


def mask_planes(planes: torch.Tensor, mask_squares: int = 8, generator: torch.Generator | None = None) -> torch.Tensor:
    masked = planes.clone()
    batch = masked.shape[0]
    squares = torch.randint(0, 64, (batch, mask_squares), generator=generator)
    for row in range(batch):
        masked[row, :, squares[row] // 8, squares[row] % 8] = 0.0
    return masked


def ssl_loss(model: nn.Module, decoder: MaskedPieceDecoder, planes: torch.Tensor) -> torch.Tensor:
    masked = mask_planes(planes)
    tokens = model.encode(masked)
    predicted = decoder(tokens)
    target = planes[:, :12].flatten(2).transpose(1, 2)
    return nn.functional.binary_cross_entropy_with_logits(predicted, target)


def run_ssl(
    model: nn.Module,
    dataset: Sequence[np.ndarray],
    cfg: TrainConfig,
    steps: int,
    device: str = "cpu",
) -> dict[str, float]:
    decoder = MaskedPieceDecoder(model.preset.d_model).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(decoder.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    model.train()
    loss_value = float("nan")
    for step in range(steps):
        batch = loop.batch_from_records(loop.batch_slice(dataset, step, cfg.batch_size))
        planes = batch["planes"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = ssl_loss(model, decoder, planes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        loss_value = float(loss.detach())
    return {"ssl_loss": loss_value, "steps": float(steps)}


def run_stage(
    model: nn.Module,
    stage: str,
    data_dir: str | Path,
    cfg: TrainConfig,
    steps: int,
    device: str = "cpu",
    counts: dict[str, int] | None = None,
) -> dict[str, float]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    if counts is not None:
        check_stage_source(stage, counts)
    dataset = ShardDataset(data_dir)
    if stage == "ssl":
        return run_ssl(model, dataset, cfg, steps, device)
    return loop.train(model, dataset, cfg, steps, device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--preset", default="p6m", choices=["p6m", "p20m"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    model = build_student(args.preset)
    cfg = TrainConfig(preset=args.preset, batch_size=args.batch_size, lr=args.lr, max_steps=args.steps)
    metrics = run_stage(model, args.stage, args.data, cfg, args.steps, args.device)
    print(metrics)
    if args.out:
        save(args.out, model, step=args.steps, meta={"stage": args.stage, "metrics": metrics})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
