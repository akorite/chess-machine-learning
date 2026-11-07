"""Round data containers and teacher reanalysis."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import chess
import numpy as np

from ..data.cache import ShardDataset
from ..data.encode import planes_from_bits

Record = np.ndarray


class ConcatDataset(Sequence[Record]):
    def __init__(self, parts: Sequence[Sequence[Record]]):
        self.parts = list(parts)
        self.ends = np.cumsum([0] + [len(part) for part in self.parts])

    def __len__(self) -> int:
        return int(self.ends[-1])

    def __getitem__(self, index: int) -> Record:
        part = int(np.searchsorted(self.ends, index, side="right") - 1)
        return self.parts[part][index - int(self.ends[part])]


def load_rounds(directories: Sequence[str | Path]) -> ConcatDataset:
    return ConcatDataset(
        [ShardDataset(directory) for directory in directories if Path(directory).exists()]
    )


def board_from_record(record: Record) -> chess.Board:
    planes = planes_from_bits(record["pieces"], record["meta"])
    board = chess.Board.empty()
    for plane in range(12):
        color = chess.WHITE if plane < 6 else chess.BLACK
        piece_type = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)[plane % 6]
        for square in range(64):
            if planes[plane, square // 8, square % 8]:
                board.set_piece_at(square, chess.Piece(piece_type, color))
    meta = record["meta"]
    board.turn = chess.WHITE if meta[2] else chess.BLACK
    board.halfmove_clock = int(meta[3])
    if meta[1]:
        board.ep_square = int(meta[1]) - 1
    for bit, right in enumerate((chess.BB_H1, chess.BB_A1, chess.BB_H8, chess.BB_A8)):
        if (int(meta[0]) >> bit) & 1:
            board.castling_rights |= right
    return board


def reanalyse(
    dataset: Sequence[Record],
    label: Callable[[chess.Board], Record],
    sample: int = 1000,
    seed: int = 0,
) -> list[Record]:
    """Re-label a sample of replay positions with the latest teacher."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(sample, len(dataset)), replace=False)
    return [label(board_from_record(np.asarray(dataset[int(index)]))) for index in indices]
