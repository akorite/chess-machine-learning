"""GigaFish depth-10 labels. Strings like "0.6", "-0.8", "#1", "#-4"."""

from __future__ import annotations

from typing import Iterator

from .sources import iter_rows


def parse_eval(text: str) -> tuple[int | None, int | None]:
    text = text.strip()
    if text.startswith("#"):
        return None, int(text[1:].lstrip("+")) if text[1:] else None
    return round(float(text) * 100), None


def iter_evals(source: str, limit_shards: int = 0) -> Iterator[dict]:
    for row in iter_rows(source, ["fen", "eval"], limit_shards):
        cp, mate = parse_eval(row["eval"])
        yield {"fen": row["fen"], "cp": cp, "mate": mate}
