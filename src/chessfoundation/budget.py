"""Refuse a GPU run that would exceed its phase or the remaining budget."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from .config import BudgetConfig

CU_PER_HOUR = {"L4": 1.94, "A100": 8.67, "G4": 10.05, "H100": 18.07}


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class BudgetState:
    remaining_cu: float
    used_cu: float = 0.0


def parse_colab_usage(text: str) -> BudgetState:
    remaining = re.search(r"([\d.]+)\s*compute units?", text, re.IGNORECASE)
    used = re.search(r"([\d.]+)\s*(?:consumed|used|spent)", text, re.IGNORECASE)
    if not remaining:
        raise BudgetExceeded("could not read a compute-unit balance from `colab usage`")
    return BudgetState(
        remaining_cu=float(remaining.group(1)),
        used_cu=float(used.group(1)) if used else 0.0,
    )


def read_colab_state(runner=subprocess.run) -> BudgetState:
    try:
        proc = runner(["colab", "usage"], capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise BudgetExceeded(f"`colab usage` unavailable: {exc}") from exc
    return parse_colab_usage(proc.stdout)


def project_cu(accel: str, hours: float) -> float:
    rate = CU_PER_HOUR.get(accel)
    if rate is None:
        raise BudgetExceeded(f"unknown accelerator {accel!r}; known: {sorted(CU_PER_HOUR)}")
    if hours <= 0:
        raise BudgetExceeded(f"hours must be positive, got {hours}")
    return rate * hours


def check_run(
    cfg: BudgetConfig,
    state: BudgetState,
    accel: str,
    hours: float,
    phase: str | None = None,
) -> float:
    """Return the projected CU or raise with the first failing bound."""
    projected = project_cu(accel, hours)
    if projected > cfg.ceiling_cu:
        raise BudgetExceeded(
            f"projected {projected:.1f} CU exceeds the {cfg.ceiling_cu:.0f} CU ceiling"
        )
    available = state.remaining_cu - cfg.reserve_cu
    if projected > available:
        raise BudgetExceeded(
            f"projected {projected:.1f} CU leaves under the {cfg.reserve_cu:.0f} CU reserve "
            f"(remaining {state.remaining_cu:.2f})"
        )
    if phase is not None:
        by_name = {p.name: p for p in cfg.phases}
        if phase not in by_name:
            raise BudgetExceeded(f"unknown phase {phase!r}")
        if projected > by_name[phase].cu:
            raise BudgetExceeded(
                f"projected {projected:.1f} CU exceeds phase {phase!r} allocation "
                f"of {by_name[phase].cu:.0f} CU"
            )
    return projected
