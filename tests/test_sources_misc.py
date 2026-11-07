import chess
import pytest

from chessfoundation.data import chessbench, gigafish, lc0_games, puzzles, syzygy


def test_gigafish_parses_centipawns_and_mate():
    assert gigafish.parse_eval("0.6") == (60, None)
    assert gigafish.parse_eval("-0.8") == (-80, None)
    assert gigafish.parse_eval("#1") == (None, 1)
    assert gigafish.parse_eval("#-4") == (None, -4)


def test_chessbench_value_to_wdl():
    assert chessbench.value_to_wdl(0.5) == 2
    assert chessbench.value_to_wdl(-0.5) == 0
    assert chessbench.value_to_wdl(0.01) == 1


def test_puzzles_filter_rating_and_length(tmp_path):
    csv = tmp_path / "puzzles.csv"
    csv.write_text(
        "PuzzleId,FEN,Moves,Rating\n"
        'a,"8/8/8/8/8/8/8/K6k w - - 0 1","a1a2",1500\n'
        'b,"8/8/8/8/8/8/8/K6k w - - 0 1","a1a2 a2a3",2900\n'
        'c,"8/8/8/8/8/8/8/K6k w - - 0 1","a1a2 a2a3 a3a4 a4a5",1500\n'
    )
    found = list(puzzles.iter_puzzles(csv))
    assert [p.puzzle_id for p in found] == ["a"]


def test_games_yield_positions(tmp_path):
    pgn = tmp_path / "games.pgn"
    pgn.write_text(
        '[Event "t"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 1-0\n\n'
        '[Event "u"]\n[Result "1/2-1/2"]\n\n1. d4 d5 1/2-1/2\n'
    )
    boards = list(lc0_games.iter_boards([pgn]))
    assert len(boards) == 6
    assert boards[0].piece_at(chess.E4) is not None


def test_syzygy_missing_table_returns_none(tmp_path):
    tablebase = syzygy.open_tablebase(str(tmp_path))
    assert syzygy.probe_wdl(chess.Board("8/8/8/8/8/8/8/KQ5k w - - 0 1"), tablebase) is None
    tablebase.close()
