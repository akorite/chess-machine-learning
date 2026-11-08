"""Pinned rung ladder: climb only while the interval excludes an even score."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
        reached = interval_excludes_even(raw, raw_low, raw_high) and result.score > 0.5
        steps.append(LadderStep(rung, result, score, low, high, reached))
        if not reached:
            break
    return steps


def best_rung(steps: list[LadderStep]) -> LadderStep | None:
    beaten = [s for s in steps if s.reached]
    return beaten[-1] if beaten else None


def book_lines(path: str, max_plies: int = 12, max_lines: int = 4096) -> tuple[str, ...]:
    """Walk a polyglot book into distinct UCI lines, deepest-first up to `max_plies`."""
    import chess
    import chess.polyglot

    if not path:
        return ()
    lines: list[str] = []
    with chess.polyglot.open_reader(str(path)) as reader:

        def walk(board: chess.Board, prefix: list[str]) -> None:
            if len(lines) >= max_lines:
                return
            entries = [] if board.ply() >= max_plies else list(reader.find_all(board))
            if not entries:
                lines.append(" ".join(prefix))
                return
            for entry in sorted(entries, key=lambda e: -e.weight):
                board.push(entry.move)
                walk(board, prefix + [entry.move.uci()])
                board.pop()

        walk(chess.Board(), [])
    return tuple(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ..engine.uci import EngineSpec, rung_options, validate_rung
    from .report import Row, render
    from .tournament import MatchConfig, run_local

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="path to the student UCI engine")
    parser.add_argument("--opponent", required=True, help="path to the pinned opponent")
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--movetime", type=int, default=50)
    parser.add_argument("--book", default="")
    parser.add_argument("--pgn", default="ladder.pgn")
    parser.add_argument("--out", default="report.md")
    args = parser.parse_args(argv)

    openings = book_lines(args.book) if args.book else ()
    engine = EngineSpec(args.engine)

    def play(rung: Rung, games: int) -> MatchResult:
        # nodes rungs pin via Limit(nodes=N) per game, not a UCI option
        options = {"Threads": 1, "Hash": 16} if rung.nodes else rung_options(rung.uci_elo or 2000)
        opponent = EngineSpec(args.opponent, options)
        probe = opponent.open()
        try:
            validate_rung(probe, options)
        finally:
            probe.quit()
        result = run_local(
            MatchConfig(
                white=engine,
                black=opponent,
                games=games,
                movetime_ms=args.movetime,
                openings=openings,
                nodes=rung.nodes,
                pgn_path=f"{args.pgn}.{rung.name}.pgn",
            )
        )
        return result

    steps = climb(play, games=args.games)
    best = best_rung(steps)
    rows = [
        Row(step.rung.describe(), step.result, f"pinned rung, {args.games} games, 1+0")
        for step in steps
    ]
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(render(rows, best.rung.name if best else None, simulations=None))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
