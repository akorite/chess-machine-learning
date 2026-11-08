import json

import numpy as np

from chessfoundation.data import cache
from chessfoundation.data.encode import RECORD_DTYPE


def records(n):
    array = np.zeros(n, dtype=RECORD_DTYPE)
    for i in range(n):
        array["wdl"][i] = i % 3
    return array


def test_shard_round_trip_and_indexing(tmp_path):
    cache.write_shard(tmp_path / "shard-00000.bin", records(3))
    cache.write_shard(tmp_path / "shard-00001.bin", records(2))
    cache.write_manifest(tmp_path, {"records": 5})
    dataset = cache.ShardDataset(tmp_path)
    assert len(dataset) == 5
    assert dataset[0]["wdl"] == 0
    assert dataset[3]["wdl"] == 0
    assert dataset[4]["wdl"] == 1


def test_verify_accepts_intact_shards(tmp_path):
    cache.write_shard(tmp_path / "shard-00000.bin", records(4))
    cache.write_manifest(tmp_path, {"records": 4})
    assert cache.verify(tmp_path) == []


def test_verify_reports_corruption(tmp_path):
    path = cache.write_shard(tmp_path / "shard-00000.bin", records(4))
    cache.write_manifest(tmp_path, {"records": 4})
    path.write_bytes(path.read_bytes()[:-1] + b"\xff")
    errors = cache.verify(tmp_path)
    assert any("checksum mismatch" in e for e in errors)


def test_verify_reports_missing_manifest(tmp_path):
    assert cache.verify(tmp_path)


def test_dataset_refuses_unverified_cache(tmp_path):
    cache.write_shard(tmp_path / "shard-00000.bin", records(2))
    try:
        cache.ShardDataset(tmp_path)
    except ValueError as exc:
        assert "verification" in str(exc)
    else:
        raise AssertionError("unverified cache must be refused")


def test_build_clears_stale_shards(tmp_path):
    cache.write_shard(tmp_path / "shard-09999.bin", records(1))
    cache.write_manifest(tmp_path, {"records": 1})
    cache.write_shard(tmp_path / "shard-00000.bin", records(2))
    cache.write_manifest(tmp_path, {"records": 2})
    assert [p.name for p in cache.shard_paths(tmp_path)] == ["shard-00000.bin", "shard-09999.bin"]


def test_verify_rejects_unlisted_shard(tmp_path):
    cache.write_shard(tmp_path / "shard-00000.bin", records(2))
    cache.write_manifest(tmp_path, {"records": 2})
    cache.write_shard(tmp_path / "shard-00001.bin", records(1))
    errors = cache.verify(tmp_path)
    assert any("unlisted shard shard-00001.bin" in e for e in errors)
    try:
        cache.ShardDataset(tmp_path)
    except ValueError as exc:
        assert "shard-00001" in str(exc)
    else:
        raise AssertionError("stray shard must be rejected")


def test_build_writes_manifest_incrementally(tmp_path):
    from chessfoundation.data import build_cache

    def source():
        for i in range(2):
            yield records(1)[0]
        yield records(1)[0]  # buffered, never flushed
        raise RuntimeError("simulated crash")

    try:
        build_cache.build(source(), tmp_path / "out", shard_size=2)
    except RuntimeError:
        pass
    else:
        raise AssertionError("crash should propagate")
    errors = cache.verify(tmp_path / "out")
    assert errors == []
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["counts"]["records"] == 2
    assert list(manifest["shards"]) == ["shard-00000.bin"]
