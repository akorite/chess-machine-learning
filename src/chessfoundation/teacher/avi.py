"""Approximate value iteration targets: one-step negamax over teacher values."""

from __future__ import annotations

from collections.abc import Callable

import chess
import numpy as np

from ..data.encode import record_from_board, set_best_move

ValueFn = Callable[[chess.Board], float]


def successor_values(board: chess.Board, value_fn: ValueFn) -> list[tuple[chess.Move, float]]:
    values: list[tuple[chess.Move, float]] = []
    for move in board.legal_moves:
        child = board.copy(stack=False)
        child.push(move)
        values.append((move, -value_fn(child)))
    return values


def avi_target(
    board: chess.Board,
    value_fn: ValueFn,
    candidates: list[chess.Move] | None = None,
) -> tuple[chess.Move, float]:
    moves = candidates if candidates is not None else list(board.legal_moves)
    if not moves:
        raise ValueError("no candidate moves")
    best_move = moves[0]
    best_value = -np.inf
    for move in moves:
        child = board.copy(stack=False)
        child.push(move)
        value = -value_fn(child)
        if value > best_value:
            best_move, best_value = move, value
    return best_move, float(best_value)


def wdl_from_value(value: float, threshold: float = 0.15) -> int:
    if value > threshold:
        return 2
    if value < -threshold:
        return 0
    return 1


def avi_record(board: chess.Board, value_fn: ValueFn, candidates: list[chess.Move] | None = None) -> np.ndarray:
    move, value = avi_target(board, value_fn, candidates)
    record = record_from_board(board)
    set_best_move(record, move)
    record["wdl"] = wdl_from_value(value)
    return record
