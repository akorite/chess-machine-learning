"""Leela Chess Zero v6 training chunks: policy visits, values, moves left.

Layout pinned to lc0's writer.h (sizeof == 8356). The 1858 policy order is not
guessable, so move decoding needs the index list, fetched at data-build time.
"""

from __future__ import annotations

import struct
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np

RECORD_SIZE = 8356
PROBABILITIES_OFFSET = 8
PROBABILITIES = 1858
PLANES_OFFSET = 7440
ROOT_Q_OFFSET = 8280
BEST_Q_OFFSET = 8284
PLIES_LEFT_OFFSET = 8304
VISITS_OFFSET = 8340
POLICY_INDEX_URL = (
    "https://raw.githubusercontent.com/Rocketknight1/minimal_lczero/master/policy_index.py"
)


def read_record(raw: bytes) -> dict:
    if len(raw) != RECORD_SIZE:
        raise ValueError(f"expected {RECORD_SIZE} bytes, got {len(raw)}")
    version, input_format = struct.unpack_from("<II", raw, 0)
    probabilities = np.frombuffer(raw, dtype="<f4", count=PROBABILITIES, offset=PROBABILITIES_OFFSET)
    return {
        "version": version,
        "input_format": input_format,
        "probabilities": probabilities.copy(),
        "planes": np.frombuffer(raw, dtype="<u8", count=104, offset=PLANES_OFFSET).copy(),
        "root_q": struct.unpack_from("<f", raw, ROOT_Q_OFFSET)[0],
        "best_q": struct.unpack_from("<f", raw, BEST_Q_OFFSET)[0],
        "plies_left": struct.unpack_from("<f", raw, PLIES_LEFT_OFFSET)[0],
        "visits": struct.unpack_from("<I", raw, VISITS_OFFSET)[0],
    }


def top_moves(probabilities: np.ndarray, policy_index: list[str], k: int = 8) -> list[tuple[str, float]]:
    order = np.argsort(probabilities)[::-1][:k]
    return [(policy_index[int(i)], float(probabilities[int(i)])) for i in order]


def fetch_policy_index(dest: str | Path) -> Path:
    path = Path(dest)
    if not path.exists():
        urllib.request.urlretrieve(POLICY_INDEX_URL, path)
    return path


def load_policy_index(path: str | Path) -> list[str]:
    import ast

    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "policy_index":
            return list(ast.literal_eval(node.value))
    raise ValueError(f"no policy_index assignment in {path}")


def iter_records(path: str | Path) -> Iterator[dict]:
    with open(path, "rb") as handle:
        while raw := handle.read(RECORD_SIZE):
            yield read_record(raw)
