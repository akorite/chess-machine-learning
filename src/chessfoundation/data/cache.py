"""Fixed-record shards: write, checksum, memmap read."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .encode import RECORD_DTYPE

MANIFEST = "manifest.json"


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
    manifest = {"counts": counts, "shards": entries}
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

    def __getitem__(self, index: int) -> np.ndarray:
        shard = int(np.searchsorted(self.ends, index, side="right") - 1)
        return np.asarray(self.shards[shard][index - int(self.ends[shard])])
