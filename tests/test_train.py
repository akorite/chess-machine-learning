import numpy as np
import torch

from chessfoundation.config import Preset, TrainConfig
from chessfoundation.data.encode import RECORD_DTYPE
from chessfoundation.model.student import Student
from chessfoundation.train import checkpoint, loop, throughput


def tiny_model():
    return Student(Preset("tiny", d_model=32, n_layers=2, n_heads=4, stem_channels=8, stem_blocks=1))


def tiny_dataset(n=32):
    records = np.zeros(n, dtype=RECORD_DTYPE)
    for i in range(n):
        records["pieces"][i] = 0
        records["meta"][i] = 0
        records["policy"][i, 0] = 12 * 64 + 28  # e2e4
        records["policy_prob"][i, 0] = 65535
        records["wdl"][i] = 2
        records["moves_left"][i] = 100
        records["promo"][i] = 255
    return records


def test_overfit_drives_policy_loss_down():
    torch.manual_seed(0)
    model = tiny_model()
    data = tiny_dataset()
    first = loop.train(model, data, TrainConfig(batch_size=8, lr=1e-2, warmup_steps=1), steps=1)
    after = loop.train(model, data, TrainConfig(batch_size=8, lr=1e-2, warmup_steps=1), steps=100)
    assert after["policy"] < first["policy"] * 0.6
    assert after["wdl"] < first["wdl"]


def test_resume_reproduces_the_next_step():
    torch.manual_seed(0)
    model = tiny_model()
    data = tiny_dataset()
    cfg = TrainConfig(batch_size=4, lr=1e-3, warmup_steps=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loop.train(model, data, cfg, steps=5, optimizer=optimizer)
    state = checkpoint.save("/tmp/ckpt.pt", model, optimizer, step=5)
    baseline = loop.train(model, data, cfg, steps=1, optimizer=optimizer, start_step=5)

    reloaded = tiny_model()
    optimizer2 = torch.optim.AdamW(reloaded.parameters(), lr=cfg.lr)
    info = checkpoint.load(state, reloaded, optimizer2)
    assert info["step"] == 5
    resumed = loop.train(reloaded, data, cfg, steps=1, optimizer=optimizer2, start_step=5)
    assert abs(resumed["loss"] - baseline["loss"]) < 1e-4


def test_learning_rate_warms_up_then_decays():
    cfg = TrainConfig(lr=1.0, warmup_steps=10, max_steps=100)
    assert loop.learning_rate(0, cfg) < loop.learning_rate(9, cfg)
    assert loop.learning_rate(10, cfg) == cfg.lr
    assert loop.learning_rate(100, cfg) < cfg.lr


def test_throughput_reports_positive_numbers():
    result = throughput.measure(tiny_model(), batch_size=4, warmup=1, steps=2)
    assert result.positions_per_second > 0
    assert result.latency_ms_batch1 > 0
    assert result.cu_projection("L4", 1000) > 0


def test_missing_targets_are_ignored():
    batch = loop.batch_from_records(list(tiny_dataset(2)))
    batch["policy_prob"][:] = 0
    batch["wdl"][:] = 255
    batch["pv"][:] = 0
    batch["moves_left"][:] = 255
    outputs = tiny_model()(batch["planes"])
    loss, metrics = loop.compute_loss(outputs, batch)
    assert torch.isfinite(loss)
    assert metrics["policy"] == 0.0
    assert metrics["wdl"] == 0.0
