"""Match report: protocol, Elo with intervals, diagnostics, negative results."""

from __future__ import annotations

from dataclasses import dataclass

from .tournament import MatchResult, elo_from_score, score_interval


@dataclass(frozen=True)
class Row:
    name: str
    result: MatchResult
    protocol: str


def render(
    rows: list[Row],
    best_rung: str | None,
    puzzle_rate: float | None = None,
    latency_ms: float | None = None,
    simulations: float | None = None,
    notes: str = "",
) -> str:
    lines = ["# Match report", "", "## Protocol", ""]
    for row in rows:
        lines.append(f"- {row.name}: {row.protocol}")
    lines.append("")
    lines.append("| Match | Games | W/D/L | Score | Elo | 95% interval |")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        score, low, high = score_interval(row.result.wins, row.result.draws, row.result.losses)
        lines.append(
            f"| {row.name} | {row.result.games} | {row.result.wins}/{row.result.draws}/"
            f"{row.result.losses} | {score:.3f} | {elo_from_score(score):+.0f} | "
            f"{elo_from_score(low):+.0f} to {elo_from_score(high):+.0f} |"
        )
    lines.append("")
    lines.append("## Diagnostics")
    lines.append("")
    if puzzle_rate is not None:
        lines.append(f"- Puzzle accuracy: {puzzle_rate:.1%}")
    if latency_ms is not None:
        lines.append(f"- Policy-only latency: {latency_ms:.2f} ms")
    if simulations is not None:
        lines.append(f"- Simulations per move: {simulations:.0f}")
    lines.append("")
    lines.append("## Result")
    lines.append("")
    if best_rung is None:
        lines.append("Negative result: the engine did not clear the L2 floor. No rung is claimed.")
    else:
        lines.append(f"Best cleared rung: {best_rung}.")
    if notes:
        lines.append("")
        lines.append(notes)
    return "\n".join(lines) + "\n"
