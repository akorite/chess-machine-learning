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


def tiny_model():
    return Student(Preset(d_model=32, n_layers=1, n_heads=4, stem_channels=8, stem_blocks=1))


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

def test_qat_checkpoint_resumes_cleanly(tmp_path):
    import torch

    from chessfoundation.data import cache
    from chessfoundation.quant.qat import prepare_qat
    from chessfoundation.train.checkpoint import save

    cache.write_shard(tmp_path / "shard-00000.bin", records(8))
    cache.write_manifest(tmp_path, {"records": 8, "24plus": 8})

    from chessfoundation.model.presets import build_student
    model = build_student("p6m")
    prepare_qat(model)
    ckpt = tmp_path / "mid_qat.pt"
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save(ckpt, model, opt, step=10, meta={"stage": "deep"})

    out = tmp_path / "resumed.pt"
    rc = pretrain.main([
        "--stage", "deep",
        "--data", str(tmp_path),
        "--preset", "p6m",
        "--steps", "2",
        "--batch-size", "4",
        "--qat",
        "--resume", str(ckpt),
        "--out", str(out),
    ])
    assert rc == 0
    assert out.exists()


def cached(tmp_path, n=40):
    from chessfoundation.data import cache

    array = records(n)
    array["meta"][:, 2] = 1
    array["cp"] = 100
    cache.write_shard(tmp_path / "shard-00000.bin", array)
    cache.write_manifest(tmp_path, {"records": n})
    return tmp_path


def metrics_of(path):
    import json

    return json.loads(path.with_suffix(".metrics.json").read_text())


def test_five_step_run_writes_validation_metrics(tmp_path):
    data = cached(tmp_path)
    out = tmp_path / "run.pt"
    rc = pretrain.main([
        "--stage", "value",
        "--data", str(data),
        "--preset", "p6m",
        "--steps", "5",
        "--batch-size", "4",
        "--out", str(out),
    ])
    assert rc == 0
    metrics = metrics_of(out)
    assert {"value_mae", "policy_top1", "wdl_accuracy", "loss", "policy"} <= set(metrics)
    assert metrics["step"] == 5
    assert np.isfinite(metrics["value_mae"])


def test_save_every_two_over_five_steps_checkpoints_twice_then_finally(tmp_path, monkeypatch):
    from chessfoundation.train import checkpoint

    data = cached(tmp_path)
    out = tmp_path / "run.pt"
    steps: list[int] = []
    real_save = checkpoint.save

    def recording_save(path, model, optimizer=None, step=0, meta=None):
        steps.append(step)
        return real_save(path, model, optimizer, step, meta)

    monkeypatch.setattr(pretrain, "save", recording_save)
    pretrain.run_stage(
        tiny_model(),
        "value",
        data,
        TrainConfig(batch_size=4, lr=1e-3, max_steps=5),
        5,
        save_path=out,
        save_every=2,
    )
    assert steps == [2, 4, 5]
    assert metrics_of(out)["step"] == 5


def test_derived_cadence_checkpoints_between_probe_and_end(tmp_path, monkeypatch):
    from chessfoundation.train import checkpoint

    data = cached(tmp_path)
    out = tmp_path / "run.pt"
    steps: list[int] = []
    real_save = checkpoint.save

    def recording_save(path, model, optimizer=None, step=0, meta=None):
        steps.append(step)
        return real_save(path, model, optimizer, step, meta)

    monkeypatch.setattr(pretrain, "save", recording_save)
    monkeypatch.setattr(pretrain, "SAVE_WINDOW", 0.001)
    pretrain.run_stage(
        tiny_model(),
        "value",
        data,
        TrainConfig(batch_size=4, lr=1e-3, max_steps=40),
        40,
        save_path=out,
    )
    assert steps[0] == pretrain.PROBE_STEPS
    assert steps[-1] == 40
    assert len(steps) > 2


def test_resume_continues_at_the_recorded_step_and_writes_metrics(tmp_path):
    import torch

    from chessfoundation.model.presets import build_student
    from chessfoundation.train import checkpoint

    data = cached(tmp_path)
    mid = tmp_path / "mid.pt"
    model = build_student("p6m")
    cfg = TrainConfig(batch_size=4, lr=1e-3, max_steps=4)
    pretrain.run_stage(
        model,
        "value",
        data,
        cfg,
        2,
        optimizer=torch.optim.AdamW(model.parameters(), lr=cfg.lr),
        save_path=mid,
        save_every=1,
    )
    assert checkpoint.load(mid, build_student("p6m"))["step"] == 2
    assert metrics_of(mid)["step"] == 2

    out = tmp_path / "resumed.pt"
    rc = pretrain.main([
        "--stage", "value",
        "--data", str(data),
        "--preset", "p6m",
        "--steps", "2",
        "--batch-size", "4",
        "--resume", str(mid),
        "--out", str(out),
    ])
    assert rc == 0
    metrics = metrics_of(out)
    assert metrics["step"] == 4
    assert {"value_mae", "policy_top1", "wdl_accuracy"} <= set(metrics)


def test_evaluate_is_deterministic_and_restores_train_mode(tmp_path):
    from chessfoundation.data.cache import ShardDataset
    from chessfoundation.train import loop

    data = ShardDataset(cached(tmp_path))
    model = tiny_model()
    cfg = TrainConfig(batch_size=4)
    first = loop.evaluate(model, data, cfg, max_batches=2)
    second = loop.evaluate(model, data, cfg, max_batches=2)
    assert first == second
    assert 0.0 <= first["policy_top1"] <= 1.0
    assert 0.0 <= first["wdl_accuracy"] <= 1.0
    assert np.isfinite(first["value_mae"])
    assert model.training
