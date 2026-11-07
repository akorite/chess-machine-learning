"""UCI engine: book, tablebase, then Gumbel search under the clock."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import chess
from torch import nn

from ..config import PRESETS
from ..model.presets import build_student
from .book import probe_book
from .gumbel import SearchConfig, SearchResult, search
from .tablebase import tablebase_move
from .time import DEFAULT_TIME, TimeConfig, allocate


class ModelEngine:
    def __init__(
        self,
        model: nn.Module,
        simulations: int = 512,
        book_path: str = "",
        syzygy_path: str = "",
        rng: random.Random | None = None,
        time_config: TimeConfig = DEFAULT_TIME,
    ):
        self.model = model.eval()
        self.simulations = simulations
        self.book_path = book_path
        self.syzygy_path = syzygy_path
        self.rng = rng or random.Random()
        self.time_config = time_config
        self.tablebase = None
        if syzygy_path:
            import chess.syzygy

            try:
                self.tablebase = chess.syzygy.open_tablebase(syzygy_path)
            except OSError:
                self.tablebase = None

    def best_move(self, board: chess.Board, time_ms: float = 400.0, ply: int = 0) -> SearchResult:
        book = probe_book(self.book_path, board, self.rng)
        if book is not None and ply < 24:
            return SearchResult(book, 0)
        probing = tablebase_move(board, self.tablebase)
        if probing is not None:
            return SearchResult(probing, 0)
        budget = allocate(time_ms, ply, self.time_config)
        # shortcut: one simulation per millisecond until throughput is measured per accelerator
        simulations = max(1, min(self.simulations, int(budget)))
        return search(self.model, board, SearchConfig(simulations=simulations))

    def run_uci(self, stdin=sys.stdin, stdout=sys.stdout) -> None:
        board = chess.Board()
        for raw in stdin:
            parts = raw.split()
            if not parts:
                continue
            command = parts[0]
            if command == "uci":
                stdout.write("id name chessfoundation\nid author chessfoundation\n")
                stdout.write("option name Simulations type spin default 512 min 1 max 100000\n")
                stdout.write("uciok\n")
            elif command == "isready":
                stdout.write("readyok\n")
            elif command == "ucinewgame":
                board = chess.Board()
            elif command == "position":
                board = _parse_position(parts)
            elif command == "go":
                remaining = _clock_for(board.turn, parts)
                result = self.best_move(board, remaining, ply=board.ply())
                stdout.write(f"info nodes {result.simulations} string {result.simulations} sims\n")
                stdout.write(f"bestmove {result.move.uci()}\n")
            elif command == "quit":
                break
            stdout.flush()


def _parse_position(parts: list[str]) -> chess.Board:
    if "startpos" in parts:
        board = chess.Board()
    elif "fen" in parts:
        start = parts.index("fen") + 1
        end = parts.index("moves") if "moves" in parts else len(parts)
        board = chess.Board(" ".join(parts[start:end]))
    else:
        board = chess.Board()
    if "moves" in parts:
        for token in parts[parts.index("moves") + 1 :]:
            board.push_uci(token)
    return board


def _clock_for(turn: chess.Color, parts: list[str]) -> float:
    if "wtime" in parts and "btime" in parts:
        key = "wtime" if turn == chess.WHITE else "btime"
        try:
            return float(parts[parts.index(key) + 1])
        except (IndexError, ValueError):
            return 400.0
    if "movetime" in parts:
        try:
            return float(parts[parts.index("movetime") + 1])
        except (IndexError, ValueError):
            return 400.0
    return 400.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--preset", default="p20m", choices=sorted(PRESETS))
    parser.add_argument("--simulations", type=int, default=512)
    parser.add_argument("--book", default="")
    parser.add_argument("--syzygy", default="")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    model = build_student(args.preset)
    if args.checkpoint:
        from ..train.checkpoint import load

        load(Path(args.checkpoint), model, map_location=args.device)
    model.to(args.device)
    engine = ModelEngine(model, args.simulations, args.book, args.syzygy)
    engine.run_uci()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
