import numpy as np
import pytest
import torch

from chessfoundation.config import Preset, TrainConfig
from chessfoundation.data.encode import RECORD_DTYPE
from chessfoundation.distill.prior_kl import masked_entropy, prior_kl
from chessfoundation.distill.replay import ConcatDataset, board_from_record
from chessfoundation.distill.rounds import label_positions, label_record, run_round
from chessfoundation.model.student import Student


def tiny_model():
    return Student(Preset("tiny", d_model=32, n_layers=1, n_heads=4, stem_channels=8, stem_blocks=1))


def records(n=8):
    array = np.zeros(n, dtype=RECORD_DTYPE)
    for i in range(n):
        array["policy"][i, 0] = 700
        array["policy_prob"][i, 0] = 65535
    return array


def test_prior_kl_is_zero_for_matching_distributions():
    logits = torch.tensor([[1.0, 2.0, 0.0]])
    mask = torch.ones_like(logits, dtype=torch.bool)
    assert float(prior_kl(logits, logits, mask)) == pytest.approx(0.0, abs=1e-6)


def test_prior_kl_is_positive_when_student_diverges():
    teacher = torch.tensor([[3.0, 0.0, 0.0]])
    student = torch.tensor([[0.0, 3.0, 0.0]])
    mask = torch.ones_like(teacher, dtype=torch.bool)
    assert float(prior_kl(teacher, student, mask)) > 0.5


def test_prior_kl_ignores_masked_actions():
    teacher = torch.tensor([[3.0, -3.0]])
    student = torch.tensor([[3.0, -3.0]])
    mask = torch.tensor([[True, False]])
    assert float(prior_kl(teacher, student, mask)) == pytest.approx(0.0, abs=1e-6)
    assert float(masked_entropy(teacher, mask)) == pytest.approx(0.0, abs=1e-6)


def test_concat_dataset_indexes_across_parts():
    first = records(3)
    second = records(2)
    dataset = ConcatDataset([first, second])
    assert len(dataset) == 5
    assert dataset[2]["policy"][0] == 700
    assert dataset[4]["policy"][0] == 700


def test_board_round_trips_through_record_planes():
    import chess

    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    from chessfoundation.data.encode import record_from_board

    rebuilt = board_from_record(record_from_board(board))
    assert rebuilt.board_fen() == board.board_fen()
    assert rebuilt.turn == board.turn
    assert rebuilt.castling_rights == board.castling_rights


def test_label_record_keeps_legal_policy_and_wdl():
    import chess

    board = chess.Board()
    policy = {chess.Move.from_uci("e2e4"): 0.7, chess.Move.from_uci("d2d4"): 0.3}
    record = label_record(board, 0.6, policy)
    assert record["policy"][0] == chess.E2 * 64 + chess.E4
    assert record["policy_prob"][0] > record["policy_prob"][1]
    assert record["wdl"] == 2


def test_label_positions_calls_analyser_per_position():
    import chess

    seen = []

    def analyse(board):
        seen.append(board.fen())
        return 0.0, {chess.Move.from_uci("e2e4"): 1.0}

    records = label_positions([chess.Board(), chess.Board()], analyse)
    assert len(records) == 2 and len(seen) == 2


def test_run_round_writes_shard_and_trains(tmp_path):
    import chess

    model = tiny_model()
    cfg = TrainConfig(batch_size=4, lr=1e-2, max_steps=3, warmup_steps=1)
    positions = [chess.Board() for _ in range(4)]

    def analyse(board):
        return 0.2, {chess.Move.from_uci("e2e4"): 1.0}

    metrics = run_round(model, positions, analyse, cfg, tmp_path / "round1")
    assert metrics["records"] == 4
    assert (tmp_path / "round1" / "shard-00000.bin").exists()
    assert np.isfinite(metrics["loss"])


def test_run_round_refuses_empty_positions(tmp_path):
    with pytest.raises(ValueError, match="no positions"):
        run_round(tiny_model(), [], lambda b: (0.0, {}), TrainConfig(), tmp_path / "r")


def test_play_and_collect_records_only_reached_positions():
    import sys
    from pathlib import Path

    from chessfoundation.distill.rounds import play_and_collect
    from chessfoundation.engine.uci import EngineSpec

    stub = [sys.executable, str(Path(__file__).with_name("stub_engine.py"))]
    white, black = EngineSpec(stub).open(), EngineSpec(stub).open()
    try:
        positions = play_and_collect(white, black, games=1, movetime_ms=2, max_plies=6)
    finally:
        white.quit()
        black.quit()
    assert 0 < len(positions) <= 6
    assert all(p.fen() for p in positions)
