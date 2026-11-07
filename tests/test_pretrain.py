import numpy as np
import pytest

from chessfoundation.budget import BudgetExceeded, BudgetState
from chessfoundation.config import BudgetConfig, Preset, TrainConfig
from chessfoundation.data.encode import RECORD_DTYPE
from chessfoundation.model.student import Student
from chessfoundation.train import pretrain
from chessfoundation.train.gates import (
    GateFailed,
    StageMetrics,
    check_stage_source,
    evaluate,
    gate,
    project_stage,
)
from chessfoundation.train.loop import batch_from_records


def tiny_model():
    return Student(Preset("tiny", d_model=32, n_layers=1, n_heads=4, stem_channels=8, stem_blocks=1))


def records(n=8):
    array = np.zeros(n, dtype=RECORD_DTYPE)
    for i in range(n):
        array["policy"][i, 0] = 100
        array["policy_prob"][i, 0] = 65535
        array["pv"][i, 0] = 100
        array["wdl"][i] = 1
        array["moves_left"][i] = 255
        array["promo"][i] = 255
    return array


def test_evaluate_reports_all_metrics():
    metrics = evaluate(tiny_model(), records(), limit=8, batch_size=4)
    assert 0.0 <= metrics.policy_accuracy <= 1.0
    assert 0.0 <= metrics.pv_accuracy <= 1.0
    assert np.isfinite(metrics.value_mae)


def test_gate_rejects_value_regression():
    previous = StageMetrics(0.5, 0.4, 0.20)
    current = StageMetrics(0.55, 0.45, 0.21)
    with pytest.raises(GateFailed, match="value MAE"):
        gate(previous, current)


def test_gate_rejects_policy_regression():
    with pytest.raises(GateFailed, match="policy accuracy"):
        gate(StageMetrics(0.5, 0.4, 0.2), StageMetrics(0.4, 0.4, 0.1))


def test_gate_accepts_improvement_and_first_stage():
    gate(None, StageMetrics(0.1, 0.1, 0.9))
    gate(StageMetrics(0.5, 0.4, 0.2), StageMetrics(0.6, 0.45, 0.15))


def test_deep_stage_refuses_shallow_rows():
    with pytest.raises(GateFailed, match="sub-depth-20"):
        check_stage_source("deep", {"24plus": 10, "lt20": 1})
    with pytest.raises(GateFailed, match="no depth"):
        check_stage_source("deep", {"lt20": 100})


def test_deep_stage_accepts_deep_rows():
    check_stage_source("deep", {"24plus": 10, "lt20": 0})


def test_projection_refuses_over_phase():
    with pytest.raises(BudgetExceeded, match="phase"):
        project_stage(BudgetConfig(), BudgetState(remaining_cu=200), "L4", 30)


def test_ssl_masking_zeroes_exactly_one_square_per_row():
    planes = np.ones((3, 20, 8, 8), dtype=np.float32)
    import torch

    masked = pretrain.mask_planes(torch.from_numpy(planes), mask_squares=1)
    for row in range(3):
        zeroed = (masked[row].flatten(1).sum(0) == 0).sum().item()
        assert zeroed == 1


def test_ssl_step_runs_and_returns_loss():
    model = tiny_model()
    cfg = TrainConfig(batch_size=4, lr=1e-2, max_steps=5, warmup_steps=1)
    metrics = pretrain.run_ssl(model, records(8), cfg, steps=3)
    assert np.isfinite(metrics["ssl_loss"])
    assert metrics["steps"] == 3
