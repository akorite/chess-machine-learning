import json

import pyarrow as pa
import pyarrow.parquet as pq

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
    pq.write_table(pa.table({"fen": [], "depth": [], "knodes": [], "cp": [], "mate": [], "line": []}), source)
    try:
        build_cache.build(build_cache.lichess_records(str(source), 24, 100_000), tmp_path / "out")
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
