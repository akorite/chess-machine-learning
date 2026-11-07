"""Syzygy probing that never raises on missing tables."""

from __future__ import annotations

import chess

MAX_PIECES = 5


def tablebase_move(board: chess.Board, tablebase) -> chess.Move | None:
    if tablebase is None or chess.popcount(board.occupied) > MAX_PIECES:
        return None
    try:
        target = tablebase.probe_wdl(board)
    except (chess.syzygy.MissingTableError, KeyError, OSError):
        return None
    best: chess.Move | None = None
    best_dtz = 0
    for move in board.legal_moves:
        board.push(move)
        try:
            value = -tablebase.probe_wdl(board)
            dtz = abs(tablebase.probe_dtz(board))
        except (chess.syzygy.MissingTableError, KeyError, OSError):
            board.pop()
            continue
        board.pop()
        if value == target and (best is None or dtz < best_dtz):
            best, best_dtz = move, dtz
    return best
