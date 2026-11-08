import math

import numpy as np
import torch

from chessfoundation.config import Preset, TrainConfig
from chessfoundation.data.encode import CP_UNKNOWN, RECORD_DTYPE
from chessfoundation.model.student import Student
from chessfoundation.train import checkpoint, loop, throughput


def tiny_model():
    return Student(Preset(d_model=32, n_layers=2, n_heads=4, stem_channels=8, stem_blocks=1))


def tiny_dataset(n=32):
    records = np.zeros(n, dtype=RECORD_DTYPE)
    for i in range(n):
        records["pieces"][i] = 0
        records["meta"][i] = 0
        records["policy"][i, 0] = 12 * 64 + 28  # e2e4
        records["policy_prob"][i, 0] = 65535
        records["wdl"][i] = 2
        records["cp"][i] = CP_UNKNOWN
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


def test_throughput_defaults_to_the_plan_batch_size():
    assert throughput.measure(tiny_model(), warmup=1, steps=1).batch_size == 512


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
    assert metrics["value"] == 0.0


def test_value_target_grows_with_centipawns_and_masks_unknown():
    records = tiny_dataset(3)
    records["meta"][:, 2] = 1  # white to move, so positive cp favours the side to move
    records["cp"][:2] = (30, 600)
    batch = loop.batch_from_records(list(records))
    assert batch["value_mask"].tolist() == [1.0, 1.0, 0.0]
    assert 0.0 < batch["value"][0] < batch["value"][1] <= 1.0




def test_compute_loss_reports_a_finite_value_term():
    records = tiny_dataset(4)
    records["cp"][:] = np.arange(4) * 100
    batch = loop.batch_from_records(list(records))
    loss, metrics = loop.compute_loss(tiny_model()(batch["planes"]), batch)
    assert math.isfinite(loss.detach())
    assert set(metrics) == {
        "loss",
        "policy",
        "promo",
        "wdl",
        "pv",
        "moves_left",
        "prior_kl",
        "value",
    }
    assert math.isfinite(metrics["value"]) and metrics["value"] > 0


def test_pv_loss_ignores_the_policy_projection():
    records = tiny_dataset(4)
    records["pv"][:, 0] = 700
    records["pv"][:, 1] = 12 * 64 + 28
    batch = loop.batch_from_records(list(records))
    torch.manual_seed(0)
    model = tiny_model().eval()
    with torch.no_grad():
        _, before = loop.compute_loss(model(batch["planes"]), batch)
        model.heads.src.weight.normal_(std=0.1)
        _, after = loop.compute_loss(model(batch["planes"]), batch)
    assert after["pv"] == before["pv"]
    assert after["policy"] != before["policy"]


def varied_dataset(n=32):
    records = tiny_dataset(n)
    records["policy"][:, 0] = np.arange(n) + 1
    records["moves_left"] = np.arange(n) % 200
    return records


def batch_order(seed, start_step, steps, n=32, batch_size=8):
    return np.concatenate(
        [loop.batch_indices(n, batch_size, seed, start_step + s) for s in range(steps)]
    )


def test_sampler_order_is_seeded():
    assert np.array_equal(batch_order(3, 0, 6), batch_order(3, 0, 6))
    assert not np.array_equal(batch_order(3, 0, 6), batch_order(4, 0, 6))


def test_epoch_covers_every_index_once():
    n, batch_size = 32, 8
    order = batch_order(1, 0, n // batch_size)
    assert np.array_equal(np.sort(order), np.arange(n))


def test_resume_replays_the_same_batch_order():
    data = varied_dataset()
    straight = list(loop.batches(data, 8, seed=2, start_step=0, steps=5))
    resumed = list(loop.batches(data, 8, seed=2, start_step=3, steps=2))
    for after, resumed_batch in zip(straight[3:], resumed):
        for key, value in after.items():
            assert torch.equal(value, resumed_batch[key])




def test_shard_stream_is_deterministic_and_covers_the_shard(tmp_path):
    from chessfoundation.data import cache

    data = varied_dataset()
    cache.write_shard(tmp_path / "shard-00000.bin", data)
    cache.write_manifest(tmp_path, {"records": len(data)})
    dataset = cache.ShardDataset(tmp_path)

    def collect(start: int, count: int) -> list[bytes]:
        return [frame.tobytes() for frame in dataset.stream(8, seed=3, start_batch=start, batches=count)]

    first = collect(0, 2)
    assert first == collect(0, 2)
    assert first != collect(1, 2)
    resumed = collect(1, 1)
    assert resumed[0] == first[1]

    rows = sorted(row.tobytes() for frame in dataset.stream(8, seed=3, batches=len(data) // 8) for row in frame)
    assert rows == sorted(row.tobytes() for row in data)


def test_stream_respects_the_training_row_limit(tmp_path):
    from chessfoundation.data import cache

    data = varied_dataset()
    cache.write_shard(tmp_path / "shard-00000.bin", data)
    cache.write_manifest(tmp_path, {"records": len(data)})
    dataset = cache.ShardDataset(tmp_path)

    limit = len(data) - 16
    rows = sum(len(frame) for frame in dataset.stream(8, seed=0, limit=limit))
    assert rows <= limit
