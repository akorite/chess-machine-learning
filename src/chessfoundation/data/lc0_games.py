"""Human game PGNs for self-supervised board pretraining."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import chess
import chess.pgn


def iter_games(paths: Iterable[str | Path]) -> Iterator[chess.pgn.Game]:
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as handle:
            while game := chess.pgn.read_game(handle):
                yield game


def iter_boards(paths: Iterable[str | Path], max_plies: int = 200) -> Iterator[chess.Board]:
    for game in iter_games(paths):
        board = game.board()
        for ply, move in enumerate(game.mainline_moves()):
            if ply >= max_plies:
                break
            board.push(move)
            yield board.copy()
