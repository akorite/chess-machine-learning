
from chessfoundation.eval.report import Row, render
from chessfoundation.eval.tournament import MatchResult


def rows():
    return [
        Row("vs L2 2000", MatchResult(250, 100, 50), "UCI_Elo 2000, 1 thread, 1+0, book on"),
        Row("vs L3 2400", MatchResult(120, 160, 120), "UCI_Elo 2400, 1 thread, 1+0, book on"),
    ]


def test_report_lists_protocol_and_intervals():
    text = render(rows(), best_rung="L3", puzzle_rate=0.8, latency_ms=9.5, simulations=5000)
    assert "UCI_Elo 2400" in text
    assert "95% interval" in text
    assert "Puzzle accuracy: 80.0%" in text
    assert "Policy-only latency: 9.50 ms" in text
    assert "Simulations per move: 5000" in text
    assert "L3" in text


def test_report_flags_a_negative_result():
    text = render([Row("vs L2 2000", MatchResult(40, 60, 300), "pinned")], best_rung=None)
    assert "Negative result" in text
    assert "L2" in text


def test_report_omits_missing_diagnostics():
    text = render(rows(), best_rung="L2")
    assert "Puzzle accuracy" not in text
    assert "latency" not in text.lower()
