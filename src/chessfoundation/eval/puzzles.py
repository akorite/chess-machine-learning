"""Puzzle accuracy: does the engine find the first solution move?"""

from __future__ import annotations

from typing import Callable, Iterable

import chess

from ..data.puzzles import Puzzle

Analyser = Callable[[chess.Board], chess.Move | None]


def solve_rate(puzzles: Iterable[Puzzle], analyse: Analyser) -> float:
    total = solved = 0
    for puzzle in puzzles:
        if len(puzzle.moves) < 2:
            continue
        board = chess.Board(puzzle.fen)
        board.push_uci(puzzle.moves[0])  # opponent's move, pre-played in the FEN
        expected = board.parse_uci(puzzle.moves[1])
        total += 1
        if analyse(board.copy()) == expected:
            solved += 1
    if total == 0:
        raise ValueError("no puzzles")
    return solved / total
