import json
import struct
from typing import ClassVar

import chess
import chess.engine
import chess.pgn
import chess.polyglot
import pytest

from chessfoundation.data.puzzles import Puzzle
from chessfoundation.engine.uci import EngineSpec
from chessfoundation.eval import ladder
from chessfoundation.eval.ladder import LADDER, best_rung, book_lines, climb
from chessfoundation.eval.puzzles import solve_rate
from chessfoundation.eval.tournament import (
    MatchConfig,
    MatchResult,
    fastchess_command,
    run_local,
)


def make_result(score):
    games = 400
    wins = round(score * games)
    return MatchResult(wins=wins, draws=0, losses=games - wins)


def test_ladder_climbs_while_results_hold():
    scores = {"L2": 0.70, "L3": 0.60, "L4": 0.45}
    steps = climb(lambda rung, games: make_result(scores.get(rung.name, 0.0)))
    assert [s.rung.name for s in steps] == ["L2", "L3", "L4"]
    assert [s.reached for s in steps] == [True, True, False]
    assert best_rung(steps).rung.name == "L3"


def test_ladder_stops_at_first_unconverted_rung():
    scores = {"L2": 0.52, "L3": 0.90}
    steps = climb(lambda rung, games: make_result(scores.get(rung.name, 0.0)))
    assert len(steps) == 1
    assert steps[0].rung.name == "L2"
    assert best_rung(steps) is None


def test_ladder_pins_documented_rungs():
    assert [r.name for r in LADDER] == ["L2", "L3", "L4"]
    assert LADDER[0].uci_elo == 2000
    assert LADDER[1].uci_elo == 2400
    assert LADDER[2].nodes == 10_000


def test_solve_rate_uses_second_puzzle_move():
    puzzle = Puzzle(
        puzzle_id="x",
        fen="8/8/8/8/8/8/8/K6k b - - 0 1",
        moves=("h1h2", "a1a2"),
        rating=1500,
    )
    import chess

    def analyse(board):
        assert board.fen().startswith("8/8/8/8/8/8/7k/K7")
        return chess.Move.from_uci("a1a2")

    assert solve_rate([puzzle], analyse) == 1.0


def test_solve_rate_reports_fraction():
    puzzles = [
        Puzzle("a", "8/8/8/8/8/8/8/K6k b - - 0 1", ("h1h2", "a1a2"), 1500),
        Puzzle("b", "8/8/8/8/8/8/8/K6k b - - 0 1", ("h1g1", "a1b1"), 1500),
    ]
    import chess

    def analyse(board):
        return chess.Move.from_uci("a1a2")

    assert solve_rate(puzzles, analyse) == pytest.approx(0.5)


def test_ladder_stops_after_a_decisive_loss():
    steps = climb(lambda rung, games: make_result(0.0))
    assert len(steps) == 1
    assert not steps[0].reached


def make_book(path, *moves):
    board = chess.Board()
    entries = []
    for uci in moves:
        move = chess.Move.from_uci(uci)
        promo = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}[
            move.promotion
        ]
        encoded = move.to_square | (move.from_square << 6) | (promo << 12)
        entries.append((chess.polyglot.zobrist_hash(board), encoded, 100, 0))
        board.push(move)
    # polyglot find_all binary-searches: entries must be key-sorted
    entries.sort(key=lambda e: e[0])
    path.write_bytes(b"".join(struct.pack(">QHHI", *e) for e in entries))


def test_book_lines_walks_book_to_leaf(tmp_path):
    book = tmp_path / "book.bin"
    make_book(book, "e2e4", "e7e5", "g1f3")
    assert book_lines(book) == ("e2e4 e7e5 g1f3",)


def test_ladder_book_wires_openings_into_match(tmp_path, monkeypatch):
    book = tmp_path / "book.bin"
    make_book(book, "e2e4", "e7e5")
    captured = []

    class FakeEngine:
        options: ClassVar[dict] = {"UCI_LimitStrength": True, "UCI_Elo": 2000, "Threads": 1, "Hash": 16}

        def configure(self, options):
            pass

        def quit(self):
            pass

    monkeypatch.setattr(EngineSpec, "open", lambda self: FakeEngine())
    monkeypatch.setattr(
        "chessfoundation.eval.tournament.run_local",
        lambda cfg, seed=0: (captured.append(cfg), MatchResult(wins=2, draws=0, losses=0))[1],
    )
    ladder.main(
        [
            "--engine",
            "student",
            "--opponent",
            "opp",
            "--games",
            "4",
            "--book",
            str(book),
            "--pgn",
            str(tmp_path / "l.pgn"),
            "--out",
            str(tmp_path / "r.md"),
        ]
    )
    assert captured
    assert captured[0].openings == ("e2e4 e7e5",)


def test_book_openings_alternate_colours_in_pgns(tmp_path, monkeypatch):
    class FirstMove:
        def play(self, board, limit):
            return chess.engine.PlayResult(next(iter(board.legal_moves)), None)

        def quit(self):
            pass

    monkeypatch.setattr(EngineSpec, "open", lambda self: FirstMove())
    cfg = MatchConfig(
        white=EngineSpec("a"),
        black=EngineSpec("b"),
        games=4,
        movetime_ms=5,
        openings=("e2e4 e7e5",),
        pgn_path=str(tmp_path / "m.pgn"),
    )
    run_local(cfg)
    games = []
    with open(tmp_path / "m.pgn") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            games.append(game)
    assert len(games) == 4
    for i, game in enumerate(games):
        moves = list(game.mainline_moves())
        assert [m.uci() for m in moves[:2]] == ["e2e4", "e7e5"]
        expected = ("white", "black") if i % 2 == 0 else ("black", "white")
        assert (game.headers["White"], game.headers["Black"]) == expected


def test_nodes_rung_uses_limit_not_option(tmp_path, monkeypatch):
    recorders = {"student": [], "opp": []}

    class Recorder:
        def __init__(self, name):
            self.name = name

        def play(self, board, limit, info=0):
            recorders[self.name].append(limit)
            return chess.engine.PlayResult(
                next(iter(board.legal_moves)), None, {"nodes": 10_000, "depth": 8}
            )

        def quit(self):
            pass

    def fake_open(self):
        return Recorder(self.path)

    monkeypatch.setattr(EngineSpec, "open", fake_open)
    cfg = MatchConfig(
        white=EngineSpec("student"),
        black=EngineSpec("opp"),
        games=2,
        movetime_ms=5,
        nodes=10_000,
        pgn_path=str(tmp_path / "m.pgn"),
    )
    run_local(cfg)
    assert any(limit.nodes == 10_000 for limit in recorders["opp"])
    assert all(limit.time is not None for limit in recorders["student"])
    assert "nodes 10000" in (tmp_path / "m.pgn").read_text()


def test_fastchess_st_is_seconds():
    cfg = MatchConfig(white=EngineSpec("a"), black=EngineSpec("b"), movetime_ms=50)
    command = fastchess_command(cfg)
    assert "st=0.05" in command


def test_pretrain_out_writes_metrics_json(tmp_path):
    import numpy as np

    from chessfoundation.data import cache
    from chessfoundation.data.encode import RECORD_DTYPE
    from chessfoundation.train import pretrain

    array = np.zeros(4, dtype=RECORD_DTYPE)
    for i in range(4):
        array["wdl"][i] = i % 3
    data = tmp_path / "data"
    cache.write_shard(data / "shard-00000.bin", array)
    cache.write_manifest(data, {"records": 4})
    out = tmp_path / "x.pt"
    pretrain.main(
        [
            "--stage",
            "ssl",
            "--data",
            str(data),
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--out",
            str(out),
        ]
    )
    metrics = out.with_suffix(".metrics.json")
    assert metrics.exists()
    saved = json.loads(metrics.read_text())
    assert saved["steps"] == 1


def test_parse_rungs_from_cli_text():
    rungs = ladder.parse_rungs("1320, 1500, nodes:10000")
    assert [(r.uci_elo, r.nodes) for r in rungs] == [(1320, None), (1500, None), (10000, None)][:2] + [(None, 10000)]
    assert ladder.parse_rungs("") == ladder.LADDER
