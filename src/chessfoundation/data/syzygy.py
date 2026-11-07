"""Syzygy probing with a missing-table sentinel."""

from __future__ import annotations

import chess
import chess.syzygy


def open_tablebase(path: str) -> chess.syzygy.Tablebase:
    return chess.syzygy.open_tablebase(path)


def probe_wdl(board: chess.Board, tablebase: chess.syzygy.Tablebase) -> int | None:
    try:
        return tablebase.probe_wdl(board)
    except (chess.syzygy.MissingTableError, KeyError):
        return None


def label_wdl(board: chess.Board, tablebase: chess.syzygy.Tablebase) -> int | None:
    wdl = probe_wdl(board, tablebase)
    if wdl is None:
        return None
    return {2: 2, 1: 1, 0: 1, -1: 0, -2: 0}[wdl]
