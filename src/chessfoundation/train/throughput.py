"""Throughput and latency measurement, converted to compute units."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn

from ..budget import CU_PER_HOUR


@dataclass(frozen=True)
class Throughput:
    positions_per_second: float
    latency_ms_batch1: float
    batch_size: int

    def cu_projection(self, accel: str, positions: int) -> float:
        hours = positions / self.positions_per_second / 3600.0
        return CU_PER_HOUR[accel] * hours


def measure(model: nn.Module, batch_size: int = 16, device: str = "cpu", warmup: int = 2, steps: int = 5) -> Throughput:
    model.eval()
    sample = torch.zeros(batch_size, 20, 8, 8, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        started = time.perf_counter()
        for _ in range(steps):
            model(sample)
        elapsed = time.perf_counter() - started
        single = torch.zeros(1, 20, 8, 8, device=device)
        model(single)
        single_started = time.perf_counter()
        for _ in range(5):
            model(single)
        single_elapsed = (time.perf_counter() - single_started) / 5
    return Throughput(
        positions_per_second=steps * batch_size / max(elapsed, 1e-9),
        latency_ms_batch1=single_elapsed * 1000.0,
        batch_size=batch_size,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ..model.presets import build_student

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="p20m", choices=["p6m", "p20m"])
    parser.add_argument("--accelerator", default="L4", choices=sorted(CU_PER_HOUR))
    parser.add_argument("--positions", type=int, default=100_000_000)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    model = build_student(args.preset)
    result = measure(model, args.batch_size)
    print(
        f"{result.positions_per_second:.1f} pos/s  "
        f"{result.latency_ms_batch1:.2f} ms/position  "
        f"{result.cu_projection(args.accelerator, args.positions):.2f} CU "
        f"for {args.positions:,} positions on {args.accelerator}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
