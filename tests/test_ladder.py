import pytest

from chessfoundation.data.puzzles import Puzzle
from chessfoundation.eval.ladder import LADDER, best_rung, climb
from chessfoundation.eval.puzzles import solve_rate
from chessfoundation.eval.tournament import MatchResult


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
