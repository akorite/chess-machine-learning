"""Clock allocation for bullet games."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeConfig:
    move_overhead_ms: float = 30.0
    expected_moves_left: int = 40
    min_ms: float = 10.0
    max_ms: float = 2000.0
    reserve_fraction: float = 0.15


def allocate(remaining_ms: float, ply: int, cfg: TimeConfig = TimeConfig()) -> float:
    """Time for this move; keeps a reserve for the endgame."""
    if remaining_ms <= 0:
        return 0.0
    spendable = remaining_ms - cfg.reserve_fraction * remaining_ms
    moves_left = max(cfg.expected_moves_left - ply // 2, 1)
    budget = spendable / moves_left
    ceiling = min(cfg.max_ms, max(remaining_ms - cfg.move_overhead_ms, cfg.min_ms))
    return max(cfg.min_ms, min(budget, ceiling))
