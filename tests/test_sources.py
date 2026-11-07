import pyarrow as pa
import pyarrow.parquet as pq

from chessfoundation.data.lichess_evals import depth_bands, is_deep, iter_evals
from chessfoundation.data.sources import iter_rows


def make_parquet(path, depth, knodes, cp, mate, line="e2e4"):
    table = pa.table(
        {
            "fen": ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * len(depth),
            "depth": [str(d) for d in depth],
            "knodes": knodes,
            "cp": cp,
            "mate": mate,
            "line": [line] * len(depth),
        }
    )
    pq.write_table(table, path)
    return path


def test_iter_rows_reads_local_shard(tmp_path):
    path = make_parquet(tmp_path / "a.parquet", [30], [100], [20], [None])
    rows = list(iter_rows(str(path)))
    assert rows[0]["depth"] == "30"


def test_deep_filter_uses_depth_or_knodes(tmp_path):
    make_parquet(
        tmp_path / "a.parquet",
        depth=[23, 23, 24],
        knodes=[10, 100_000, 10],
        cp=[10, 20, 30],
        mate=[None, None, None],
    )
    rows = list(iter_evals(str(tmp_path), min_depth=24, min_knodes=100_000))
    assert len(rows) == 2
    assert rows[0].knodes == 100_000
    assert rows[1].depth == 24


def test_plain_rows_kept_when_not_deep(tmp_path):
    make_parquet(tmp_path / "a.parquet", [10], [10], [0], [None])
    assert len(list(iter_evals(str(tmp_path), deep=False))) == 1


def test_is_deep_boundaries():
    assert is_deep({"depth": "24", "knodes": 0}, 24, 100_000)
    assert is_deep({"depth": "10", "knodes": 100_000}, 24, 100_000)
    assert not is_deep({"depth": "23", "knodes": 99_999}, 24, 100_000)


def test_depth_bands_report_counts(tmp_path):
    make_parquet(
        tmp_path / "a.parquet",
        depth=[10, 21, 24, 30],
        knodes=[1, 1, 1, 100_001],
        cp=[0, 0, 0, 0],
        mate=[None] * 4,
    )
    assert depth_bands(str(tmp_path)) == {
        "lt20": 1,
        "20to23": 1,
        "24plus": 2,
        "knodes100k": 1,
    }


def test_local_directory_respects_limit(tmp_path):
    make_parquet(tmp_path / "a.parquet", [30], [1], [0], [None])
    make_parquet(tmp_path / "b.parquet", [30], [1], [0], [None])
    assert len(list(iter_rows(str(tmp_path), limit_shards=1))) == 1
