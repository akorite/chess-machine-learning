import struct

import numpy as np
import pytest

from chessfoundation.data import lc0_chunks


def make_record(probabilities, root_q, best_q, plies_left, visits):
    raw = bytearray(lc0_chunks.RECORD_SIZE)
    struct.pack_into("<II", raw, 0, 6, 3)
    for i, p in enumerate(probabilities):
        struct.pack_into("<f", raw, lc0_chunks.PROBABILITIES_OFFSET + 4 * i, p)
    struct.pack_into("<f", raw, lc0_chunks.ROOT_Q_OFFSET, root_q)
    struct.pack_into("<f", raw, lc0_chunks.BEST_Q_OFFSET, best_q)
    struct.pack_into("<f", raw, lc0_chunks.PLIES_LEFT_OFFSET, plies_left)
    struct.pack_into("<I", raw, lc0_chunks.VISITS_OFFSET, visits)
    return bytes(raw)


def policy_index():
    index = ["a1a1"] * lc0_chunks.PROBABILITIES
    index[7] = "e2e4"
    index[11] = "d2d4"
    return index


def test_read_record_decodes_documented_fields():
    raw = make_record([0.0] * 1858, 0.25, 0.5, 42.0, 800)
    record = lc0_chunks.read_record(raw)
    assert record["version"] == 6
    assert record["input_format"] == 3
    assert record["root_q"] == pytest.approx(0.25)
    assert record["best_q"] == pytest.approx(0.5)
    assert record["plies_left"] == pytest.approx(42.0)
    assert record["visits"] == 800


def test_probabilities_sum_and_length():
    probabilities = np.zeros(1858, dtype=np.float32)
    probabilities[7] = 0.75
    probabilities[11] = 0.25
    record = lc0_chunks.read_record(make_record(probabilities, 0, 0, 0, 0))
    assert record["probabilities"].shape == (1858,)
    assert record["probabilities"].sum() == pytest.approx(1.0)


def test_top_moves_returns_descending_visits():
    probabilities = np.zeros(1858, dtype=np.float32)
    probabilities[7] = 0.75
    probabilities[11] = 0.25
    top = lc0_chunks.top_moves(probabilities, policy_index(), k=2)
    assert top == [("e2e4", pytest.approx(0.75)), ("d2d4", pytest.approx(0.25))]


def test_rejects_wrong_record_size():
    with pytest.raises(ValueError, match="8356"):
        lc0_chunks.read_record(b"\x00" * 10)


def test_load_policy_index_reads_python_list(tmp_path):
    path = tmp_path / "policy_index.py"
    path.write_text('policy_index = ["e2e4", "d2d4"]\n')
    assert lc0_chunks.load_policy_index(path) == ["e2e4", "d2d4"]
