"""ChessBench action-value shards. Column names are configurable per release."""

from __future__ import annotations

from collections.abc import Iterator

from .sources import iter_rows

DEFAULT_COLUMNS = {"fen": "fen", "move": "move", "value": "value"}


def value_to_wdl(value: float) -> int:
    if value > 0.15:
        return 2
    if value < -0.15:
        return 0
    return 1


def iter_actions(
    source: str,
    columns: dict[str, str] | None = None,
    limit_shards: int = 0,
) -> Iterator[dict]:
    names = columns or DEFAULT_COLUMNS
    yield from iter_rows(source, list(names.values()), limit_shards)
