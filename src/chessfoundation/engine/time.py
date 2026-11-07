"""Clock allocation for bullet games."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeConfig:
    move_overhead_ms: float = 30.0
    expected_moves_left: int = 40
    target_move_ms: float = 400.0
    min_ms: float = 10.0
    max_ms: float = 2000.0
    reserve_fraction: float = 0.15


DEFAULT_TIME = TimeConfig()


def allocate(remaining_ms: float, ply: int, cfg: TimeConfig = DEFAULT_TIME) -> float:
    """Time for this move; keeps a reserve for the endgame."""
    if remaining_ms <= 0:
        return 0.0
    spendable = remaining_ms - cfg.reserve_fraction * remaining_ms
    moves_left = max(cfg.expected_moves_left - ply // 2, 1)
    budget = min(cfg.target_move_ms, spendable / moves_left)
    ceiling = min(cfg.max_ms, remaining_ms - cfg.move_overhead_ms)
    return max(0.0, min(budget, ceiling))
