"""Parquet shard discovery and streaming, local or Hugging Face."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq


def resolve_shards(source: str, limit: int = 0) -> list[Path]:
    path = Path(source)
    if path.exists():
        files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    else:
        from huggingface_hub import HfApi, hf_hub_download

        names = [
            name
            for name in HfApi().list_repo_files(source, repo_type="dataset")
            if name.endswith(".parquet")
        ]
        names = sorted(names)
        if limit:
            names = names[:limit]
        files = [Path(hf_hub_download(source, name, repo_type="dataset")) for name in names]
    if limit and not path.exists():
        files = files[:limit]
    return files


def iter_rows(
    source: str, columns: list[str] | None = None, limit_shards: int = 0
) -> Iterator[dict]:
    for shard in resolve_shards(source, limit_shards):
        parquet = pq.ParquetFile(shard)
        available = set(parquet.schema_arrow.names)
        if columns is not None:
            missing = sorted(set(columns) - available)
            if missing:
                raise ValueError(f"{shard} is missing columns {missing}")
        wanted = columns
        for batch in parquet.iter_batches(batch_size=2048, columns=wanted):
            yield from batch.to_pylist()
