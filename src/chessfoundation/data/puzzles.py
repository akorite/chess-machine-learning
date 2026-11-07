"""Lichess puzzle ingestion and filtering."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Puzzle:
    puzzle_id: str
    fen: str
    moves: tuple[str, ...]
    rating: int


def iter_puzzles(
    path: str | Path,
    min_rating: int = 1200,
    max_rating: int = 2200,
    max_moves: int = 3,
    limit: int = 0,
) -> Iterator[Puzzle]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        emitted = 0
        for row in reader:
            try:
                rating = int(row["Rating"])
                moves = tuple(row["Moves"].split())
            except (KeyError, ValueError):
                continue
            if not (min_rating <= rating <= max_rating) or not 1 <= len(moves) <= max_moves:
                continue
            yield Puzzle(row["PuzzleId"], row["FEN"], moves, rating)
            emitted += 1
            if limit and emitted >= limit:
                return
