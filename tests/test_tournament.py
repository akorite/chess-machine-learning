import math

import chess
import pytest

from chessfoundation.eval.tournament import (
    MatchResult,
    elo_from_score,
    interval_excludes_even,
    parse_fastchess,
    score_interval,
)


def test_elo_from_known_scores():
    assert elo_from_score(1.0) == math.inf
    assert elo_from_score(0.0) == -math.inf
    assert elo_from_score(0.5) == pytest.approx(0.0)
    assert elo_from_score(0.75) == pytest.approx(190.85, abs=0.01)


def test_sixty_forty_is_seventy_elo():
    score, low, high = score_interval(60, 0, 40)
    assert elo_from_score(score) == pytest.approx(70.4, abs=0.1)
    assert interval_excludes_even(score, low, high)
    assert elo_from_score(low) > 0


def test_fifty_one_percent_over_four_hundred_does_not_exclude_even():
    score, low, high = score_interval(204, 0, 196)
    assert score == pytest.approx(0.51)
    assert not interval_excludes_even(score, low, high)
    assert elo_from_score(low) < 0 < elo_from_score(high)


def test_match_result_score_and_elo():
    result = MatchResult(wins=30, draws=40, losses=30)
    assert result.games == 100
    assert result.score == pytest.approx(0.5)
    assert result.elo == pytest.approx(0.0)


def test_parse_fastchess_summary():
    text = "Results of white vs black: 12 wins, 3 draws, 5 losses\n"
    assert parse_fastchess(text) == (12, 3, 5)


def test_parse_fastchess_rejects_unknown_output():
    with pytest.raises(ValueError):
        parse_fastchess("no summary")


def test_score_interval_needs_games():
    with pytest.raises(ValueError):
        score_interval(0, 0, 0)
