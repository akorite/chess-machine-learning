"""Fixed-record shards: write, checksum, memmap read."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .encode import RECORD_DTYPE

MANIFEST = "manifest.json"
FORMAT_STAMP = f"record-v{RECORD_DTYPE.itemsize}"  # dtype changes must invalidate old caches


_PERMUTATION: dict[tuple[int, int, int], np.ndarray] = {}


def permutation(n: int, seed: int, epoch: int) -> np.ndarray:
    """Shuffled index order for one epoch; identical arguments give identical order.

    One slot is memoized: a step asks for the same epoch's order many times, and
    regenerating an n-element permutation dominates batch prep at large n.
    """
    key = (n, seed, epoch)
    cached = _PERMUTATION.get(key)
    if cached is None:
        cached = np.random.default_rng([seed, epoch]).permutation(n)
        _PERMUTATION.clear()
        _PERMUTATION[key] = cached
    return cached


def shard_paths(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("shard-*.bin"))


def write_shard(path: str | Path, records: np.ndarray) -> Path:
    path = Path(path)
    array = np.ascontiguousarray(records, dtype=RECORD_DTYPE)
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)
    return path


def read_shard(path: str | Path) -> np.ndarray:
    path = Path(path)
    data = np.fromfile(path, dtype=RECORD_DTYPE)
    if data.size == 0:
        raise ValueError(f"empty shard {path}")
    return data


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(directory: str | Path, counts: dict[str, int]) -> Path:
    directory = Path(directory)
    entries = {
        path.name: {"sha256": sha256(path), "records": path.stat().st_size // RECORD_DTYPE.itemsize}
        for path in shard_paths(directory)
    }
    manifest = {"format": FORMAT_STAMP, "counts": counts, "shards": entries}
    path = directory / MANIFEST
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def clear_shards(directory: str | Path) -> None:
    directory = Path(directory)
    for path in shard_paths(directory):
        path.unlink()
    manifest = directory / MANIFEST
    if manifest.exists():
        manifest.unlink()


def _safe_name(name: str) -> bool:
    return Path(name).name == name and name not in (".", "..")


def verify(directory: str | Path) -> list[str]:
    directory = Path(directory)
    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        return [f"missing {manifest_path}"]
    manifest = json.loads(manifest_path.read_text())
    errors = []
    if manifest.get("format") != FORMAT_STAMP:
        errors.append(
            f"format mismatch: {manifest.get('format')!r} != {FORMAT_STAMP!r}; rebuild the cache"
        )
    for name, entry in manifest["shards"].items():
        if not _safe_name(name):
            errors.append(f"unsafe shard name {name}")
            continue
        path = directory / name
        if not path.exists():
            errors.append(f"missing shard {name}")
        elif sha256(path) != entry["sha256"]:
            errors.append(f"checksum mismatch {name}")
    for path in shard_paths(directory):
        if path.name not in manifest["shards"]:
            errors.append(f"unlisted shard {path.name}")
    return errors


class ShardDataset:
    """Random access over one or more record shards."""

    def __init__(self, directory: str | Path):
        self.shards = [np.memmap(path, dtype=RECORD_DTYPE, mode="r") for path in shard_paths(directory)]
        if not self.shards:
            raise ValueError(f"no shards in {directory}")
        problems = verify(directory)
        if problems:
            raise ValueError(f"cache {directory} failed verification: {problems}")
        self.ends = np.cumsum([0] + [len(s) for s in self.shards])

    def __len__(self) -> int:
        return int(self.ends[-1])

    def stream(
        self,
        batch_size: int,
        seed: int,
        start_batch: int = 0,
        batches: int = 0,
        limit: int = 0,
    ):
        """Batches in shuffled shard order, gathered inside one shard at a time.

        A 100M-record corpus is far larger than page cache, so a globally random
        gather turns every batch into reads across every shard file. Streaming
        shard by shard keeps each batch inside one already-open mapping.
        """
        produced = 0
        skipped = 0
        rows = 0
        epoch = 0
        while not batches or produced < batches:
            order = permutation(len(self.shards), seed, epoch)
            for shard_pos in order:
                shard = self.shards[int(shard_pos)]
                indices = permutation(len(shard), seed, epoch * 7919 + int(shard_pos))
                for start in range(0, len(indices) - batch_size + 1, batch_size):
                    if limit and rows >= limit:
                        return
                    if skipped < start_batch:
                        skipped += 1
                        continue
                    yield np.asarray(shard[indices[start : start + batch_size]])
                    produced += 1
                    rows += batch_size
                    if batches and produced >= batches:
                        return
            epoch += 1

    def __getitem__(self, index: int) -> np.ndarray:
        shard = int(np.searchsorted(self.ends, index, side="right") - 1)
        return np.asarray(self.shards[shard][index - int(self.ends[shard])])

    def batch(self, indices: np.ndarray) -> np.ndarray:
        """Gather many records at once, touching each shard with one slice."""
        indices = np.asarray(indices, dtype=np.int64)
        shard = np.searchsorted(self.ends, indices, side="right") - 1
        order = np.argsort(shard, kind="stable")
        cuts = np.searchsorted(shard[order], np.arange(len(self.shards) + 1))
        out = np.empty(len(indices), dtype=RECORD_DTYPE)
        for k, shard_data in enumerate(self.shards):
            rows = order[cuts[k] : cuts[k + 1]]
            if rows.size:
                out[rows] = shard_data[indices[rows] - self.ends[k]]
        return out
