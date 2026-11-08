import math
import time

import chess
import chess.engine
import pytest

from chessfoundation.teacher.avi import (
    avi_record,
    avi_target,
    successor_values,
    wdl_from_value,
)
from chessfoundation.teacher.lc0_oracle import policy_from_scores, value_from_score


def material_value(board: chess.Board) -> float:
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    score = 0
    for piece_type, value in values.items():
        score += value * (len(board.pieces(piece_type, chess.WHITE)) - len(board.pieces(piece_type, chess.BLACK)))
    sign = 1 if board.turn == chess.WHITE else -1
    return max(-1.0, min(1.0, sign * score / 10.0))


def test_value_from_score_converts_centipawns_and_mate():
    score = chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE)
    assert value_from_score(score, chess.WHITE) == pytest.approx(0.0)
    win = chess.engine.PovScore(chess.engine.Mate(1), chess.WHITE)
    assert value_from_score(win, chess.WHITE) == 1.0
    assert value_from_score(win, chess.BLACK) == -1.0


def test_value_from_score_is_side_to_move_relative():
    score = chess.engine.PovScore(chess.engine.Cp(300), chess.WHITE)
    assert value_from_score(score, chess.WHITE) > 0.4
    assert value_from_score(score, chess.BLACK) < -0.4


def test_policy_from_scores_normalises_multipv():
    infos = [
        {"pv": [chess.Move.from_uci("e2e4")], "score": chess.engine.PovScore(chess.engine.Cp(30), chess.WHITE)},
        {"pv": [chess.Move.from_uci("d2d4")], "score": chess.engine.PovScore(chess.engine.Cp(10), chess.WHITE)},
    ]
    policy = policy_from_scores(infos)
    assert sum(policy.values()) == pytest.approx(1.0)
    assert policy[chess.Move.from_uci("e2e4")] > policy[chess.Move.from_uci("d2d4")]


def test_successor_values_negates_child_perspective():
    board = chess.Board("7k/8/8/8/8/8/8/K6Q w - - 0 1")
    values = successor_values(board, material_value)
    assert values
    assert all(-1.0 <= v <= 1.0 for _, v in values)


def test_avi_prefers_capturing_a_hanging_queen():
    board = chess.Board("7k/8/8/8/8/8/4q2R/K7 w - - 0 1")
    move, _value = avi_target(board, material_value)
    assert move == chess.Move.from_uci("h2e2")


def test_avi_with_candidate_list_restricts_search():
    board = chess.Board()
    candidates = [chess.Move.from_uci("a2a3"), chess.Move.from_uci("e2e4")]
    move, _ = avi_target(board, material_value, candidates)
    assert move in candidates


def test_avi_record_carries_policy_and_wdl():
    board = chess.Board()
    record = avi_record(board, material_value)
    assert record["policy_prob"][0] == 65535
    assert record["wdl"] in (0, 1, 2)


def test_wdl_threshold():
    assert wdl_from_value(0.5) == 2
    assert wdl_from_value(-0.5) == 0
    assert wdl_from_value(0.0) == 1


def test_policy_from_scores_is_side_to_move_relative():
    infos = [
        {"pv": [chess.Move.from_uci("e2e4")], "score": chess.engine.PovScore(chess.engine.Cp(500), chess.WHITE)},
        {"pv": [chess.Move.from_uci("d2d4")], "score": chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE)},
    ]
    white = policy_from_scores(infos, chess.WHITE)
    black = policy_from_scores(infos, chess.BLACK)
    assert white[chess.Move.from_uci("e2e4")] > white[chess.Move.from_uci("d2d4")]
    assert black[chess.Move.from_uci("e2e4")] < black[chess.Move.from_uci("d2d4")]


def test_policy_from_scores_survives_mate_scores():
    infos = [
        {"pv": [chess.Move.from_uci("e2e4")], "score": chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)},
        {"pv": [chess.Move.from_uci("d2d4")], "score": chess.engine.PovScore(chess.engine.Cp(20), chess.WHITE)},
    ]
    policy = policy_from_scores(infos, chess.WHITE)
    assert sum(policy.values()) == pytest.approx(1.0)
    assert all(not math.isnan(value) for value in policy.values())


def test_avi_record_marks_promo_unknown_for_quiet_moves():
    record = avi_record(chess.Board(), lambda board: 0.0)
    assert record["promo"] == 255


def test_oracle_options_exclude_managed_multipv(monkeypatch):
    from chessfoundation.teacher.lc0_oracle import Lc0Oracle, StockfishOracle

    captured: list[dict] = []

    class FakeEngine:
        def configure(self, options):
            captured.append(options)

        def quit(self):
            pass

    monkeypatch.setattr(
        "chessfoundation.teacher.lc0_oracle.chess.engine.SimpleEngine.popen_uci",
        staticmethod(lambda path: FakeEngine()),
    )

    with Lc0Oracle(path="x", weights="w.pb.gz", multipv=8):
        pass
    with StockfishOracle(path="x", multipv=8):
        pass

    assert captured == [{"WeightsFile": "w.pb.gz"}]  # empty dict skips configure entirely
    assert all("MultiPV" not in opts for opts in captured)


def test_oracle_timeout_raises_typed_error():
    from chessfoundation.teacher.lc0_oracle import Lc0Oracle, OracleTimeout

    class Stalled:
        def analyse(self, board, limit, multipv=1):
            time.sleep(1.0)

    oracle = Lc0Oracle(timeout_s=0.05)
    oracle.engine = Stalled()
    with pytest.raises(OracleTimeout):
        oracle.analyse(chess.Board())


def test_oracle_empty_infos_raises_with_fen():
    from chessfoundation.teacher.lc0_oracle import OracleError, StockfishOracle

    class Empty:
        def analyse(self, board, limit, multipv=1):
            return []

    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    oracle = StockfishOracle()
    oracle.engine = Empty()
    with pytest.raises(OracleError, match=board.fen()):
        oracle.analyse(board)


def test_oracle_missing_score_raises_with_fen():
    from chessfoundation.teacher.lc0_oracle import Lc0Oracle, OracleError

    class Scoreless:
        def analyse(self, board, limit, multipv=1):
            return [{"pv": [chess.Move.from_uci("e2e4")]}]

    board = chess.Board()
    oracle = Lc0Oracle()
    oracle.engine = Scoreless()
    with pytest.raises(OracleError, match=board.fen()):
        oracle.analyse(board)


def test_oracle_parses_valid_infos():
    from chessfoundation.teacher.lc0_oracle import StockfishOracle

    score = chess.engine.PovScore(chess.engine.Cp(120), chess.WHITE)

    class Healthy:
        def analyse(self, board, limit, multipv=1):
            return [{"pv": [chess.Move.from_uci("e2e4")], "score": score, "nodes": 500}]

    oracle = StockfishOracle()
    oracle.engine = Healthy()
    result = oracle.analyse(chess.Board())
    assert result.move == chess.Move.from_uci("e2e4")
    assert result.value > 0
    assert result.nodes == 500


def test_label_positions_skips_oracle_failures(capsys):
    from chessfoundation.distill.rounds import label_positions
    from chessfoundation.teacher.lc0_oracle import OracleError

    calls = {"n": 0}

    def analyse(board):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OracleError(f"oracle stalled: {board.fen()}")
        return 0.25, {next(iter(board.legal_moves)): 1.0}

    records = label_positions([chess.Board(), chess.Board(), chess.Board()], analyse)
    assert len(records) == 2
    assert calls["n"] == 3
    assert "skipped" in capsys.readouterr().err
