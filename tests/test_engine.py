import os
import random
import struct
import subprocess
import sys
from pathlib import Path

import chess
import chess.polyglot
import chess.syzygy
import numpy as np
import pytest
import torch
from torch import nn

from chessfoundation.config import Preset
from chessfoundation.data.encode import POLICY_ACTIONS
from chessfoundation.engine.book import probe_book
from chessfoundation.engine.gumbel import SearchConfig, search
from chessfoundation.engine.model_engine import ModelEngine
from chessfoundation.engine.tablebase import tablebase_move
from chessfoundation.engine.time import TimeConfig, allocate
from chessfoundation.model.student import Student

REPO = Path(__file__).resolve().parent.parent


def tiny_model():
    return Student(Preset("tiny", d_model=32, n_layers=1, n_heads=4, stem_channels=8, stem_blocks=1))


class ScriptedModel(nn.Module):
    """Policy points at one move index; value is a draw."""

    def __init__(self, move: chess.Move):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.index = move.from_square * 64 + move.to_square

    def forward(self, planes):
        batch = planes.shape[0]
        policy = torch.zeros(batch, POLICY_ACTIONS)
        policy[:, self.index] = 10.0
        wdl = torch.zeros(batch, 3)
        wdl[:, 1] = 10.0
        return {
            "policy": policy + self.bias,
            "promo": torch.zeros(batch, 4),
            "wdl": wdl,
            "pv": torch.zeros(batch, 6, POLICY_ACTIONS),
            "moves_left": torch.zeros(batch),
        }


def test_search_returns_legal_move_within_simulation_cap():
    model = tiny_model()
    result = search(model, chess.Board(), SearchConfig(simulations=4), np.random.default_rng(0))
    assert result.move in chess.Board().legal_moves
    assert result.simulations <= 4


def test_policy_only_finds_mate_in_one():
    board = chess.Board("6k1/5ppp/8/8/8/7Q/8/K7 w - - 0 1")
    mate = chess.Move.from_uci("h3h7")  # not mate; choose a real mate below
    mate = next(m for m in board.legal_moves if chess.Board(board.fen()).is_legal(m) and _is_mate(board, m))
    model = ScriptedModel(mate)
    result = search(model, board, SearchConfig(simulations=1))
    assert result.move == mate


def _is_mate(board, move):
    child = board.copy()
    child.push(move)
    return child.is_checkmate()


def test_single_legal_move_short_circuits():
    board = chess.Board("7k/8/6K1/8/8/8/8/8 b - - 0 1")
    result = search(tiny_model(), board, SearchConfig(simulations=32))
    assert result.move in board.legal_moves
    assert result.simulations == 1


def test_time_manager_keeps_reserve_and_respects_ceiling():
    cfg = TimeConfig()
    first = allocate(60_000, 0, cfg)
    assert 0 < first <= cfg.max_ms
    assert allocate(500, 40, cfg) <= 500 - cfg.move_overhead_ms
    assert allocate(0, 0, cfg) == 0.0


def test_time_budget_shrinks_as_clock_shrinks():
    cfg = TimeConfig()
    assert allocate(60_000, 10, cfg) > allocate(5_000, 10, cfg)


def make_book(path, board, move):
    key = chess.polyglot.zobrist_hash(board)
    promo = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}[move.promotion]
    encoded = move.to_square | (move.from_square << 6) | (promo << 12)
    path.write_bytes(struct.pack(">QHHI", key, encoded, 100, 0))


def test_book_probe_returns_stored_move(tmp_path):
    path = tmp_path / "book.bin"
    make_book(path, chess.Board(), chess.Move.from_uci("e2e4"))
    assert probe_book(path, chess.Board()) == chess.Move.from_uci("e2e4")


def test_book_probe_without_file_returns_none(tmp_path):
    assert probe_book(tmp_path / "missing.bin", chess.Board()) is None


def test_tablebase_move_without_tables_returns_none(tmp_path):
    assert tablebase_move(chess.Board("8/8/8/8/8/8/8/KQ5k w - - 0 1"), None) is None
    tablebase = chess.syzygy.open_tablebase(str(tmp_path))
    try:
        assert tablebase_move(chess.Board("8/8/8/8/8/8/8/KQ5k w - - 0 1"), tablebase) is None
    finally:
        tablebase.close()


def test_engine_uses_book_move_before_search(tmp_path):
    path = tmp_path / "book.bin"
    make_book(path, chess.Board(), chess.Move.from_uci("d2d4"))
    engine = ModelEngine(tiny_model(), simulations=2, book_path=str(path))
    result = engine.best_move(chess.Board(), time_ms=100)
    assert result.move == chess.Move.from_uci("d2d4")
    assert result.simulations == 0


def test_uci_handshake_and_legal_move():
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    proc = subprocess.run(
        [sys.executable, "-m", "chessfoundation.engine.model_engine", "--preset", "p6m", "--simulations", "1"],
        input="uci\nisready\nposition startpos movetime 100\ngo movetime 100\nquit\n",
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
        timeout=120,
    )
    output = proc.stdout
    assert "uciok" in output
    assert "readyok" in output
    best = [line for line in output.splitlines() if line.startswith("bestmove")][0].split()[1]
    assert chess.Move.from_uci(best) in chess.Board().legal_moves
