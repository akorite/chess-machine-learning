"""Distill round driver: end-to-end round, reanalysis, prior-KL, round gate."""

import json
import os
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chessfoundation.config import Preset
from chessfoundation.data.encode import RECORD_DTYPE, record_from_board
from chessfoundation.distill.replay import reanalyse
from chessfoundation.distill.rounds import apply_gate, label_record
from chessfoundation.model.student import Student

SRC = str(Path(__file__).resolve().parent.parent / "src")


@pytest.fixture
def subprocess_path(monkeypatch):
    """`python -m chessfoundation...` needs src on PYTHONPATH (pytest's pythonpath is in-process only)."""
    monkeypatch.setenv("PYTHONPATH", SRC + os.pathsep + os.environ.get("PYTHONPATH", ""))


def tiny_model():
    return Student(Preset(d_model=32, n_layers=1, n_heads=4, stem_channels=8, stem_blocks=1))


# A UCI stub that answers `go` (bestmove or analyse) with multipv info lines.
# Serves as collection opponent, gate opponent, and teacher oracle in tests.
ANALYSER_STUB = '''
import sys
import chess

board = chess.Board()
for raw in sys.stdin:
    parts = raw.split()
    if not parts:
        continue
    cmd = parts[0]
    if cmd == "uci":
        print("id name analyser")
        print("option name MultiPV type spin default 4 min 1 max 500")
        print("uciok")
    elif cmd == "isready":
        print("readyok")
    elif cmd == "ucinewgame":
        board = chess.Board()
    elif cmd == "position":
        if "startpos" in parts:
            board = chess.Board()
        elif "fen" in parts:
            start = parts.index("fen") + 1
            end = parts.index("moves") if "moves" in parts else len(parts)
            board = chess.Board(" ".join(parts[start:end]))
        if "moves" in parts:
            for token in parts[parts.index("moves") + 1:]:
                board.push_uci(token)
    elif cmd == "go":
        moves = list(board.legal_moves)
        while len(moves) < 4:
            moves += moves
        for i, move in enumerate(moves[:4]):
            print("info multipv %d depth 1 score cp %d nodes 100 pv %s" % (i + 1, 40 - i * 10, move.uci()))
        print("bestmove " + moves[0].uci())
    elif cmd == "quit":
        break
    sys.stdout.flush()
'''

CRASH_STUB = '''
import sys
count = 0
for raw in sys.stdin:
    if raw.split()[:1] == ["uci"]:
        print("id name crash")
        print("uciok")
    elif raw.split()[:1] == ["isready"]:
        print("readyok")
    elif raw.split()[:1] == ["go"]:
        count += 1
        if count > 2:
            sys.exit(1)  # die mid-game; the round must still finish
        print("bestmove e2e4")
    elif raw.split()[:1] == ["quit"]:
        break
    sys.stdout.flush()
'''


def write_stub(tmp_path, source):
    path = tmp_path / f"stub_{abs(hash(source))}.py"
    path.write_text(source)
    return [sys.executable, str(path)]


def test_analyser_stub_end_to_end_round(tmp_path):
    """python -m chessfoundation.distill wires play -> label -> retrain -> checkpoint."""
    from chessfoundation.distill.__main__ import main

    stub = write_stub(tmp_path, ANALYSER_STUB)
    out = tmp_path / "run"
    rc = main([
        "--white", *stub,
        "--black", *stub,
        "--teacher-engine", *stub,
        "--teacher-depth", "1",
        "--games", "2",
        "--movetime-ms", "10",
        "--steps", "2",
        "--out", str(out),
    ])
    assert rc == 0
    records = json.loads((out / "rounds.json").read_text())
    assert len(records) == 1 and records[0]["accepted"]
    round_dir = out / records[0]["dir"]
    assert (round_dir / "shard-00000.bin").exists()
    assert records[0]["metrics"]["records"] > 0
    assert (out / "current.pt").exists()


def test_collection_survives_opponent_crash(tmp_path):
    """play_and_collect runs through play_game: an engine death forfeits, not hangs."""
    from chessfoundation.distill.rounds import play_and_collect
    from chessfoundation.engine.uci import EngineSpec

    crash = write_stub(tmp_path, CRASH_STUB)
    analyser = write_stub(tmp_path, ANALYSER_STUB)
    white, black = EngineSpec(crash).open(), EngineSpec(analyser).open()
    try:
        positions = play_and_collect(white, black, games=1, movetime_ms=10, max_plies=8)
    finally:
        try:
            white.quit()
        except chess.engine.EngineError:
            pass
        black.quit()
    assert positions  # partial game still yields its positions


def test_reanalyse_replaces_labels_for_moved_values():
    dataset = np.stack([record_from_board(chess.Board(fen)) for fen in [
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "rnbqkb1r/pp1ppppp/5n2/2pP4/8/8/PPP1PPPP/RNBQKB1R w - c6 0 3",
    ]])

    def moved(board):
        return label_record(board, 0.9, {chess.Move.from_uci("e2e4"): 1.0})

    fresh = reanalyse(dataset, moved, sample=2, seed=0)
    assert len(fresh) == 2
    assert all(int(r["wdl"]) == 2 for r in fresh)
    assert all(int(r["wdl"]) != int(old["wdl"]) or int(r["policy_prob"][0]) != int(old["policy_prob"][0])
               for r, old in zip(fresh, dataset))


def test_reanalyse_keeps_record_when_oracle_fails():
    from chessfoundation.teacher.lc0_oracle import OracleError

    dataset = np.stack([record_from_board(chess.Board())])

    def broken(board):
        raise OracleError(f"engine failed: {board.fen()}")

    fresh = reanalyse(dataset, broken, sample=1)
    assert len(fresh) == 1
    assert np.array_equal(np.asarray(fresh[0]), np.asarray(dataset[0]))


def test_prior_kl_zero_when_student_matches_teacher_prior():
    from chessfoundation.train.loop import batch_from_records, compute_loss

    array = np.zeros(2, dtype=RECORD_DTYPE)
    array["policy"][:, 0] = 700
    array["policy"][:, 1] = 800
    array["policy_prob"][:, 0] = 49151
    array["policy_prob"][:, 1] = 16384
    batch = batch_from_records(array)
    teacher_logits = torch.log(batch["policy_prob"].float().clamp(min=1e-9))
    support = torch.zeros(2, 4096)
    support.scatter_(1, batch["policy_idx"], teacher_logits)
    outputs = {
        "policy": support,
        "promo": torch.zeros(2, 4),
        "wdl": torch.zeros(2, 3),
        "pv": torch.zeros(2, 6, 4096),
        "moves_left": torch.zeros(2),
        "value": torch.zeros(2),
    }
    _, metrics = compute_loss(outputs, batch, prior_kl_weight=0.05)
    assert metrics["prior_kl"] == pytest.approx(0.0, abs=1e-5)
    outputs["policy"] = outputs["policy"].clone()
    outputs["policy"][:, int(batch["policy_idx"][0, 0])] = 4.0
    _, diverged = compute_loss(outputs, batch, prior_kl_weight=0.05)
    assert diverged["prior_kl"] > 0.0


def test_apply_gate_rejects_regression_and_keeps_current(tmp_path):
    out = tmp_path / "run"
    current = out / "current.pt"
    round_ckpt = out / "round-0001" / "checkpoint.pt"
    current.parent.mkdir(parents=True)
    round_ckpt.parent.mkdir(parents=True)
    current.write_bytes(b"previous")
    round_ckpt.write_bytes(b"regressed")
    history = [{"round": 0, "score": 0.6, "accepted": True, "dir": "round-0000"}]

    record = apply_gate(out, 1, round_ckpt, 0.5, history)
    assert not record["accepted"]
    assert current.read_bytes() == b"previous"
    assert not round_ckpt.exists()
    history.append(record)
    (out / "rounds.json").write_text(json.dumps(history))

    # next round at the previous accepted score is held and becomes current
    better_ckpt = out / "round-0002" / "checkpoint.pt"
    better_ckpt.parent.mkdir(parents=True)
    better_ckpt.write_bytes(b"held")
    record = apply_gate(out, 2, better_ckpt, 0.6, history)
    assert record["accepted"]
    assert current.read_bytes() == b"held"


def test_apply_gate_accepts_first_score(tmp_path):
    out = tmp_path / "run"
    ckpt = out / "round-0000" / "checkpoint.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"first")
    record = apply_gate(out, 0, ckpt, 0.4, [])
    assert record["accepted"]
    assert (out / "current.pt").read_bytes() == b"first"


def test_gate_match_scores_student(tmp_path, subprocess_path):
    from chessfoundation.distill.rounds import gate_match
    from chessfoundation.engine.uci import EngineSpec

    analyser = write_stub(tmp_path, ANALYSER_STUB)
    student = [sys.executable, "-m", "chessfoundation.engine.model_engine",
               "--preset", "p6m", "--simulations", "4"]
    score = gate_match(EngineSpec(student), EngineSpec(analyser), games=2,
                       movetime_ms=10, pgn_path=str(tmp_path / "gate.pgn"))
    assert 0.0 <= score <= 1.0


def test_main_gate_drop_writes_rejected_record(tmp_path):
    """A gated round that cannot match records a rejection and keeps current.pt."""
    from chessfoundation.distill.__main__ import main

    stub = write_stub(tmp_path, ANALYSER_STUB)
    out = tmp_path / "run"
    out.mkdir()
    (out / "current.pt").write_bytes(b"keep-me")
    (out / "rounds.json").write_text(json.dumps(
        [{"round": 0, "score": 1.0, "accepted": True, "dir": "round-0000"}]))
    rc = main([
        "--white", *stub,
        "--black", *stub,
        "--teacher-engine", *stub,
        "--teacher-depth", "1",
        "--gate-engine", str(tmp_path / "no-such-engine"),
        "--gate-games", "2",
        "--games", "1",
        "--movetime-ms", "10",
        "--steps", "1",
        "--out", str(out),
    ])
    assert rc == 0
    records = json.loads((out / "rounds.json").read_text())
    assert records[-1]["accepted"] is False
    assert records[-1]["gate_error"]
    assert (out / "current.pt").read_bytes() == b"keep-me"
    assert not (out / "round-0001" / "checkpoint.pt").exists()
