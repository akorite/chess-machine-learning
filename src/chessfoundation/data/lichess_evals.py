"""Streaming Lichess position evaluations with depth discipline."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .sources import iter_rows

COLUMNS = ["fen", "cp", "mate", "line", "depth", "knodes"]


@dataclass(frozen=True)
class EvalRow:
    fen: str
    cp: int | None
    mate: int | None
    line: str
    depth: int
    knodes: int


def is_deep(row: dict, min_depth: int, min_knodes: int) -> bool:
    return int(row["depth"] or 0) >= min_depth or int(row["knodes"] or 0) >= min_knodes


def iter_evals(
    source: str,
    min_depth: int = 24,
    min_knodes: int = 100_000,
    deep: bool = True,
    limit_shards: int = 0,
) -> Iterator[EvalRow]:
    for row in iter_rows(source, COLUMNS, limit_shards):
        if row.get("cp") is None and row.get("mate") is None:
            continue
        if deep and not is_deep(row, min_depth, min_knodes):
            continue
        yield EvalRow(
            fen=row["fen"],
            cp=None if row["cp"] is None else int(row["cp"]),
            mate=None if row["mate"] is None else int(row["mate"]),
            line=row["line"] or "",
            depth=int(row["depth"] or 0),
            knodes=int(row["knodes"] or 0),
        )


def depth_bands(source: str, limit_shards: int = 0) -> dict[str, int]:
    bands = {"lt20": 0, "20to23": 0, "24plus": 0, "knodes100k": 0}
    for row in iter_rows(source, ["depth", "knodes"], limit_shards):
        depth = int(row["depth"] or 0)
        if depth < 20:
            bands["lt20"] += 1
        elif depth < 24:
            bands["20to23"] += 1
        else:
            bands["24plus"] += 1
        if int(row["knodes"] or 0) >= 100_000:
            bands["knodes100k"] += 1
    return bands
