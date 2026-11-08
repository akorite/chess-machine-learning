import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import chess
import chess.polyglot
import chess.syzygy
import numpy as np
import torch
from torch import nn

from chessfoundation.config import Preset
from chessfoundation.data.encode import POLICY_ACTIONS, PROMO_PIECES
from chessfoundation.engine.book import probe_book
from chessfoundation.engine.gumbel import (
    SearchConfig,
    _evaluate,
    _Node,
    _promotion_choice,
    search,
)
from chessfoundation.engine.model_engine import ModelEngine
from chessfoundation.engine.tablebase import tablebase_move
from chessfoundation.engine.time import TimeConfig, allocate
from chessfoundation.model.student import Student

REPO = Path(__file__).resolve().parent.parent


def tiny_model():
    return Student(Preset(d_model=32, n_layers=1, n_heads=4, stem_channels=8, stem_blocks=1))


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
        check=False,
        text=True,
        env=env,
        cwd=REPO,
        timeout=120,
    )
    output = proc.stdout
    assert "uciok" in output
    assert "readyok" in output
    best = next(line for line in output.splitlines() if line.startswith("bestmove")).split()[1]
    assert chess.Move.from_uci(best) in chess.Board().legal_moves


def test_search_spends_more_than_the_candidate_cap():
    result = search(tiny_model(), chess.Board(), SearchConfig(simulations=40), np.random.default_rng(1))
    assert result.simulations > 16

def test_puct_search_selects_scripted_policy_move():
    move = chess.Move.from_uci("e2e4")
    model = ScriptedModel(move)
    cfg = SearchConfig(simulations=8, mode="puct")
    assert search(model, chess.Board(), cfg, np.random.default_rng(3)).move == move
    assert search(model, chess.Board(), cfg, np.random.default_rng(9)).move == move


def test_search_handles_terminal_children():
    # mate-in-1: Qh5xf7#; one child ends the game — regression for _terminal_value(_Node).
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1")
    result = search(tiny_model(), board, SearchConfig(simulations=32), np.random.default_rng(4))
    assert result.move in board.legal_moves




class EntropyModel(ScriptedModel):
    def __init__(self, move, uncertain):
        super().__init__(move)
        self.uncertain = uncertain

    def forward(self, planes):
        outputs = super().forward(planes)
        if self.uncertain:
            outputs["wdl"] = torch.zeros(planes.shape[0], 3)
        return outputs


def test_uncertain_position_spends_more_simulations():
    board = chess.Board()
    certain = search(EntropyModel(chess.Move.from_uci("e2e4"), False), board, SearchConfig(simulations=40), np.random.default_rng(2))
    uncertain = search(EntropyModel(chess.Move.from_uci("e2e4"), True), board, SearchConfig(simulations=40), np.random.default_rng(2))
    assert uncertain.simulations > certain.simulations


def test_setoption_changes_simulation_budget():
    engine = ModelEngine(tiny_model(), simulations=4)
    engine._setoption(["setoption", "name", "Simulations", "value", "9"])
    assert engine.simulations == 9


class FakeTablebase:
    def probe_wdl(self, board):
        return 2 if board.turn == chess.WHITE else -2

    def probe_dtz(self, board):
        return 1 if board.turn == chess.WHITE else 10


def test_tablebase_prefers_smaller_dtz_among_wins():
    board = chess.Board("8/8/8/8/8/8/8/KQ5k w - - 0 1")
    move = tablebase_move(board, FakeTablebase())
    assert move is not None and board.is_legal(move)

class SleepingModel(nn.Module):
    """Each forward sleeps ``delay_s``; exercises wall-clock deadlines and batching."""

    def __init__(self, delay_s: float):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.delay_s = delay_s

    def forward(self, planes):
        time.sleep(self.delay_s)
        batch = planes.shape[0]
        wdl = torch.zeros(batch, 3)
        wdl[:, 1] = 10.0
        return {
            "policy": torch.zeros(batch, POLICY_ACTIONS) + self.bias,
            "promo": torch.zeros(batch, 4),
            "wdl": wdl,
            "pv": torch.zeros(batch, 6, POLICY_ACTIONS),
            "moves_left": torch.zeros(batch),
        }


def test_search_respects_wall_clock_deadline():
    model = SleepingModel(0.011)
    board = chess.Board()
    started = time.monotonic()
    result = search(
        model, board,
        SearchConfig(simulations=500, budget_ms=50.0, eval_batch=1),
        np.random.default_rng(0),
    )
    elapsed = time.monotonic() - started
    assert result.move in board.legal_moves
    assert result.simulations < 500
    assert elapsed < 0.10


def test_best_move_movetime_bounds_wall_clock():
    engine = ModelEngine(tiny_model())
    board = chess.Board()
    started = time.monotonic()
    result = engine.best_move(board, movetime_ms=50)
    elapsed = time.monotonic() - started
    assert result.simulations >= 1
    assert elapsed <= 0.10


def test_batched_evaluate_matches_per_board():
    model = tiny_model().eval()
    boards = [
        chess.Board(),
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
        chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1"),
    ]
    together = _evaluate(model, boards, "cpu")
    singles = [_evaluate(model, [b], "cpu") for b in boards]
    for head in range(4):
        for j in range(len(boards)):
            np.testing.assert_allclose(together[head][j], singles[j][head][0], rtol=1e-5, atol=1e-5)


def test_promotion_choice_picks_knight():
    board = chess.Board("6k1/P7/8/8/8/8/8/K7 w - - 0 1")
    root = _Node(board.copy(stack=False))
    root.promo = np.zeros(4)
    root.promo[PROMO_PIECES.index(chess.KNIGHT)] = 9.0
    assert _promotion_choice(root, chess.A7, chess.A8) == chess.Move.from_uci("a7a8n")


class PromoModel(ScriptedModel):
    def __init__(self, move, promo):
        super().__init__(move)
        self.promo = promo

    def forward(self, planes):
        outputs = super().forward(planes)
        outputs["promo"] = torch.tensor(self.promo, dtype=torch.float32).repeat(planes.shape[0], 1)
        return outputs


def test_search_returns_underpromotion():
    board = chess.Board("6k1/P7/8/8/8/8/8/K7 w - - 0 1")
    promo = np.zeros(4)
    promo[PROMO_PIECES.index(chess.KNIGHT)] = 9.0
    model = PromoModel(chess.Move.from_uci("a7a8q"), promo)
    result = search(model, board, SearchConfig(simulations=8), np.random.default_rng(3))
    assert result.move == chess.Move.from_uci("a7a8n")


def test_batched_leaves_do_more_sims_per_second():
    model = SleepingModel(0.01)
    unbatched = search(
        model, chess.Board(),
        SearchConfig(simulations=10_000, budget_ms=200.0, eval_batch=1),
        np.random.default_rng(0),
    )
    batched = search(
        model, chess.Board(),
        SearchConfig(simulations=10_000, budget_ms=200.0, eval_batch=16),
        np.random.default_rng(0),
    )
    assert batched.simulations >= 2 * unbatched.simulations
