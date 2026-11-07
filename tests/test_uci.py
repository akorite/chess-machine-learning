import sys
from pathlib import Path

import pytest

from chessfoundation.engine.uci import (
    EngineSpec,
    nodes_options,
    rung_options,
    validate_rung,
)
from chessfoundation.eval.tournament import MatchConfig, run_local

STUB = [sys.executable, str(Path(__file__).with_name("stub_engine.py"))]


def stub_spec():
    return EngineSpec(path=STUB)


def test_rung_options_pin_limit_strength():
    assert rung_options(2400)["UCI_LimitStrength"] is True
    assert nodes_options(10_000)["Nodes"] == 10_000


def test_engine_exposes_rung_options():
    engine = stub_spec().open()
    try:
        validate_rung(engine, rung_options(2400))
        validate_rung(engine, nodes_options(10_000))
    finally:
        engine.quit()


def test_missing_option_is_rejected():
    engine = stub_spec().open()
    try:
        with pytest.raises(ValueError, match="does not expose"):
            validate_rung(engine, {"Skill Level": 20})
    finally:
        engine.quit()


def test_local_match_plays_two_games_and_writes_pgn(tmp_path):
    pgn = tmp_path / "match.pgn"
    cfg = MatchConfig(
        white=stub_spec(),
        black=stub_spec(),
        games=2,
        movetime_ms=10,
        pgn_path=str(pgn),
    )
    result = run_local(cfg)
    assert result.games == 2
    assert pgn.exists()
    assert pgn.read_text().count("[Event") == 2
