"""Stage A self-supervised masking, then value and deep-label stages."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..budget import read_colab_state
from ..config import RunConfig, TrainConfig
from ..data.cache import MANIFEST, ShardDataset
from ..model.presets import build_student
from ..quant.qat import convert, prepare_qat
from . import loop
from .checkpoint import load, save
from .gates import check_stage_source, project_stage

STAGES = ("ssl", "value", "deep")


class MaskedPieceDecoder(nn.Module):
    """Predict piece planes from masked tokens."""

    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Linear(dim, 12)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(tokens)


def mask_planes(planes: torch.Tensor, mask_squares: int = 8, generator: torch.Generator | None = None) -> torch.Tensor:
    batch = planes.shape[0]
    squares = torch.randint(0, 64, (batch, mask_squares), device=planes.device, generator=generator)
    keep = torch.ones(batch, 64, dtype=planes.dtype, device=planes.device)
    keep.scatter_(1, squares, 0.0)
    return planes * keep.view(batch, 1, 8, 8)


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
    model = model.to(device)
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
    optimizer: torch.optim.Optimizer | None = None,
    start_step: int = 0,
    save_path: str | Path | None = None,
    save_every: int = 0,
) -> dict[str, float]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    if counts is not None:
        check_stage_source(stage, counts)
    dataset = ShardDataset(data_dir)
    if stage == "ssl":
        return run_ssl(model, dataset, cfg, steps, device)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    done = 0
    metrics: dict[str, float] = {}
    while done < steps:
        chunk = min(save_every or (steps - done), steps - done)
        metrics = loop.train(model, dataset, cfg, chunk, device, optimizer, start_step + done)
        done += chunk
        if save_path:
            save(save_path, model, optimizer, step=start_step + done, meta={"stage": stage})
    metrics["step"] = float(start_step + done)
    return metrics


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
    parser.add_argument("--init", default="", help="checkpoint to start from")
    parser.add_argument("--resume", default="", help="checkpoint to resume training")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--accelerator", default="")
    parser.add_argument("--hours", type=float, default=0.0)
    parser.add_argument("--phase", default="pretrain")
    parser.add_argument("--qat", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args(argv)

    model = build_student(args.preset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    start_step = 0
    if args.init:
        load(args.init, model)
    if args.resume:
        start_step = int(load(args.resume, model, optimizer)["step"])
    if args.accelerator:
        project_stage(
            RunConfig().budget, read_colab_state(), args.accelerator, args.hours, args.phase
        )
    counts = None
    manifest = Path(args.data) / MANIFEST
    if manifest.exists():
        counts = json.loads(manifest.read_text()).get("counts")
    if args.qat:
        prepare_qat(model)
    cfg = TrainConfig(
        batch_size=args.batch_size, lr=args.lr, max_steps=args.steps + start_step,
        compile=args.compile,
    )
    metrics = run_stage(
        model, args.stage, args.data, cfg, args.steps, args.device, counts,
        optimizer, start_step, args.out or None, args.save_every,
    )
    if args.qat:
        convert(model)
    print(metrics)
    if args.out:
        save(args.out, model, optimizer, step=start_step + args.steps, meta={"stage": args.stage, "metrics": metrics})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
