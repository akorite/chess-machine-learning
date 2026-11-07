"""Pinned rung ladder: climb only while the interval excludes an even score."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .tournament import (
    MatchResult,
    elo_from_score,
    interval_excludes_even,
    score_interval,
)


@dataclass(frozen=True)
class Rung:
    name: str
    uci_elo: int | None = None
    nodes: int | None = None

    def describe(self) -> str:
        if self.nodes is not None:
            return f"{self.name}: {self.nodes} nodes/move"
        return f"{self.name}: UCI_Elo {self.uci_elo}"


LADDER: tuple[Rung, ...] = (
    Rung("L2", uci_elo=2000),
    Rung("L3", uci_elo=2400),
    Rung("L4", nodes=10_000),
)


@dataclass(frozen=True)
class LadderStep:
    rung: Rung
    result: MatchResult
    score: float
    low: float
    high: float
    reached: bool


def climb(
    play: Callable[[Rung, int], MatchResult],
    games: int = 400,
    rungs: tuple[Rung, ...] = LADDER,
) -> list[LadderStep]:
    steps: list[LadderStep] = []
    for rung in rungs:
        result = play(rung, games)
        raw, raw_low, raw_high = score_interval(result.wins, result.draws, result.losses)
        score = elo_from_score(raw)
        low = elo_from_score(raw_low)
        high = elo_from_score(raw_high)
        reached = interval_excludes_even(raw, raw_low, raw_high)
        steps.append(LadderStep(rung, result, score, low, high, reached and result.score > 0.5))
        if not reached:
            break
    return steps


def best_rung(steps: list[LadderStep]) -> LadderStep | None:
    beaten = [s for s in steps if s.reached]
    return beaten[-1] if beaten else None
