"""Polyglot opening book probing."""

from __future__ import annotations

import random
from pathlib import Path

import chess
import chess.polyglot


def probe_book(path: str | Path, board: chess.Board, rng: random.Random | None = None) -> chess.Move | None:
    if not path or not Path(path).exists():
        return None
    rng = rng or random.Random()
    try:
        with chess.polyglot.open_reader(str(path)) as reader:
            entries = list(reader.find_all(board))
    except (OSError, KeyError):
        return None
    if not entries:
        return None
    total = sum(entry.weight for entry in entries) or len(entries)
    pick = rng.uniform(0, total)
    for entry in entries:
        pick -= entry.weight or 1
        if pick <= 0:
            return entry.move
    return entries[-1].move
