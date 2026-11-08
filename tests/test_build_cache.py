import itertools
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from chessfoundation.data import build_cache, cache


def write_evals(tmp_path):
    path = tmp_path / "evals.parquet"
    table = pa.table(
        {
            "fen": [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            ],
            "depth": ["30", "30"],
            "knodes": [200_000, 200_000],
            "cp": [120, 120],
            "mate": [None, None],
            "line": ["e2e4 e7e5", "e7e5 g1f3"],
        }
    )
    pq.write_table(table, path)
    return path


def test_build_from_local_evals_produces_verified_shards(tmp_path):
    source = write_evals(tmp_path)
    out = tmp_path / "cache"
    counts = {}
    result = build_cache.build(
        build_cache.lichess_records(str(source), 24, 100_000, counts=counts),
        out,
        shard_size=1,
        counts=counts,
    )
    assert counts["24plus"] == 2
    assert counts["knodes100k"] == 2
    assert result == {"records": 2, "shards": 2}
    assert cache.verify(out) == []
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["counts"]["24plus"] == 2
    assert manifest["counts"]["knodes100k"] == 2
    dataset = cache.ShardDataset(out)
    assert len(dataset) == 2
    assert int(dataset[0]["policy"][0]) > 0
    assert dataset[0]["wdl"] == 2
    assert dataset[1]["wdl"] == 0


def test_build_refuses_empty_input(tmp_path):
    source = tmp_path / "empty.parquet"
    pq.write_table(
        pa.table(
            {"fen": [], "depth": [], "knodes": [], "cp": [], "mate": [], "line": []}
        ),
        source,
    )
    try:
        build_cache.build(
            build_cache.lichess_records(str(source), 24, 100_000), tmp_path / "out"
        )
    except ValueError as exc:
        assert "no records" in str(exc)
    else:
        raise AssertionError("empty build must fail")


def test_build_append_preserves_existing_shards(tmp_path):
    source = write_evals(tmp_path)
    out = tmp_path / "cache"
    build_cache.build(
        build_cache.lichess_records(str(source), 24, 100_000),
        out,
        shard_size=1,
    )
    assert len(cache.shard_paths(out)) == 2
    result = build_cache.build(
        build_cache.lichess_records(str(source), 24, 100_000),
        out,
        shard_size=1,
        append=True,
    )
    assert result == {"records": 4, "shards": 4}
    assert len(cache.shard_paths(out)) == 4
    assert cache.verify(out) == []
    dataset = cache.ShardDataset(out)
    assert len(dataset) == 4


FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def write_lichess_shard(path, count, cp=100):
    pq.write_table(
        pa.table(
            {
                "fen": [FEN] * count,
                "depth": ["30"] * count,
                "knodes": [200_000] * count,
                "cp": [cp] * count,
                "mate": [None] * count,
                "line": ["e2e4 e7e5 g1f3"] * count,
            }
        ),
        path,
    )
    return path


def write_gigafish_shard(path, count, evaluation="0.5"):
    pq.write_table(pa.table({"fen": [FEN] * count, "eval": [evaluation] * count}), path)
    return path


def two_source_dirs(tmp_path):
    lichess_dir = tmp_path / "lichess"
    lichess_dir.mkdir()
    write_lichess_shard(lichess_dir / "a.parquet", 3)
    write_lichess_shard(lichess_dir / "b.parquet", 3)
    gigafish_dir = tmp_path / "gigafish"
    gigafish_dir.mkdir()
    write_gigafish_shard(gigafish_dir / "a.parquet", 3)
    return lichess_dir, gigafish_dir


def record_bytes(out):
    return [
        record.tobytes()
        for path in cache.shard_paths(out)
        for record in cache.read_shard(path)
    ]


def test_cli_mix_caps_each_source_and_counts_sum(tmp_path):
    lichess_dir, gigafish_dir = two_source_dirs(tmp_path)
    out = tmp_path / "cache"
    code = build_cache.main(
        [
            "--mix",
            "lichess:4",
            "--mix",
            "gigafish:3",
            "--source-path",
            f"lichess={lichess_dir}",
            "--source-path",
            f"gigafish={gigafish_dir}",
            "--out",
            str(out),
            "--shard-size",
            "4",
            "--workers",
            "2",
        ]
    )
    assert code == 0
    counts = json.loads((out / "manifest.json").read_text())["counts"]
    assert counts["source:lichess"] == 4
    assert counts["source:gigafish"] == 3
    assert counts["records"] == 7
    assert counts["source:lichess"] + counts["source:gigafish"] == counts["records"]
    assert len(record_bytes(out)) == 7
    assert cache.verify(out) == []


def test_mix_keeps_records_from_both_sources(tmp_path):
    lichess_dir, gigafish_dir = two_source_dirs(tmp_path)
    out = tmp_path / "cache"
    result = build_cache.build_mix(
        [("lichess", 0), ("gigafish", 0)],
        out,
        named={"lichess": str(lichess_dir), "gigafish": str(gigafish_dir)},
        workers=1,
        shard_size=2,
    )
    assert result == {"records": 9, "shards": 5}
    dataset = cache.ShardDataset(out)
    assert {int(dataset[i]["cp"]) for i in range(len(dataset))} == {50, 100}


def test_parallel_build_matches_serial_build(tmp_path):
    lichess_dir, gigafish_dir = two_source_dirs(tmp_path)
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    named = {"lichess": str(lichess_dir), "gigafish": str(gigafish_dir)}
    build_cache.build(
        itertools.chain(
            build_cache.lichess_records(str(lichess_dir), 24, 100_000),
            build_cache.gigafish_records(str(gigafish_dir)),
        ),
        serial,
        shard_size=2,
    )
    result = build_cache.build_mix(
        [("lichess", 0), ("gigafish", 0)],
        parallel,
        named=named,
        workers=2,
        shard_size=2,
    )
    assert result == {"records": 9, "shards": 5}
    assert record_bytes(parallel) == record_bytes(serial)
    assert cache.verify(parallel) == []


def test_interrupted_parallel_build_leaves_verifiable_manifest(tmp_path, monkeypatch):
    lichess_dir, gigafish_dir = two_source_dirs(tmp_path)
    out = tmp_path / "cache"
    original = build_cache._Writer.add
    calls = []

    def add_then_fail(self, block):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("interrupted")
        return original(self, block)

    monkeypatch.setattr(build_cache._Writer, "add", add_then_fail)
    with pytest.raises(RuntimeError):
        build_cache.build_mix(
            [("lichess", 0), ("gigafish", 0)],
            out,
            named={"lichess": str(lichess_dir), "gigafish": str(gigafish_dir)},
            workers=1,
            shard_size=1,
        )
    assert calls
    assert cache.verify(out) == []
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["counts"]["records"] == len(record_bytes(out)) > 0


def test_verify_rejects_stray_shard_after_mix(tmp_path):
    lichess_dir, _ = two_source_dirs(tmp_path)
    out = tmp_path / "cache"
    build_cache.build_mix(
        [("lichess", 2)],
        out,
        named={"lichess": str(lichess_dir)},
        workers=1,
        shard_size=2,
    )
    (out / "shard-00099.bin").write_bytes(b"stray")
    assert any("unlisted shard shard-00099.bin" in e for e in cache.verify(out))
