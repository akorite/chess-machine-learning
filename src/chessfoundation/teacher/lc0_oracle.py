"""Teacher oracles: Lc0 and Stockfish through python-chess's UCI client."""

from __future__ import annotations

import math
import queue
import threading
from dataclasses import dataclass, field
from typing import Self

import chess
import chess.engine


@dataclass(frozen=True)
class OracleResult:
    move: chess.Move
    value: float
    pv: tuple[chess.Move, ...] = ()
    policy: dict[chess.Move, float] = field(default_factory=dict)
    nodes: int = 0



class OracleError(RuntimeError):
    """Teacher failure; carries the board FEN so the position can be skipped."""


class OracleTimeout(OracleError, TimeoutError):
    """The engine did not answer analyse inside the bounded wait."""


def value_from_score(score: chess.engine.PovScore, turn: chess.Color) -> float:
    pov = score.pov(turn)
    if pov.is_mate():
        mate = pov.mate()
        return 1.0 if mate and mate > 0 else -1.0
    cp = pov.score(mate_score=100_000)
    if cp is None:
        return 0.0
    return 2.0 / (1.0 + math.exp(-cp / 300.0)) - 1.0


def policy_from_scores(
    infos: list[chess.engine.InfoDict], turn: chess.Color = chess.WHITE, temperature: float = 120.0
) -> dict[chess.Move, float]:
    raw: dict[chess.Move, float] = {}
    for info in infos:
        pv = info.get("pv") or ()
        score = info.get("score")
        if not pv or score is None:
            continue
        pov = score.pov(turn)
        cp = pov.score(mate_score=100_000) or 0
        raw[pv[0]] = max(-1000.0, min(1000.0, cp)) / temperature
    if not raw:
        return {}
    peak = max(raw.values())
    weights = {move: math.exp(value - peak) for move, value in raw.items()}
    total = sum(weights.values())
    return {move: weight / total for move, weight in weights.items()}


class UciOracle:
    """Base for engines that answer through UCI analyse."""

    def __init__(self, path: str, options: dict | None = None, multipv: int = 4, timeout_s: float = 30.0):
        self.path = path
        self.timeout_s = timeout_s
        self.options = options or {}
        self.multipv = multipv
        self.engine: chess.engine.SimpleEngine | None = None

    def open(self) -> UciOracle:
        self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
        if self.options:
            self.engine.configure(dict(self.options))
        return self

    def close(self) -> None:
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def _analyse(self, board: chess.Board, limit: chess.engine.Limit) -> OracleResult:
        assert self.engine is not None, "oracle not open"
        infos = self._call_engine(board, limit)
        usable = [info for info in infos if isinstance(info, dict) and info.get("pv")]
        if not usable:
            raise OracleError(f"engine returned no pv lines: {board.fen()}")
        policy = policy_from_scores(usable, board.turn)
        best = usable[0]
        score = best.get("score")
        if not isinstance(score, chess.engine.PovScore) or score.pov(board.turn).score(mate_score=100_000) is None:
            raise OracleError(f"engine returned no usable score: {board.fen()}")
        pv = tuple(best["pv"])
        return OracleResult(
            move=pv[0],
            value=value_from_score(score, board.turn),
            pv=pv,
            policy=policy,
            nodes=int(best.get("nodes") or 0),
        )

    def _call_engine(self, board: chess.Board, limit: chess.engine.Limit) -> list[chess.engine.InfoDict]:
        # SimpleEngine.analyse has no timeout kwarg; bound the wait in a daemon
        # thread (a dead thread leaks but cannot wedge the run) and let
        # Limit(time=...) make healthy engines return before the cutoff.
        limit.time = min(limit.time, self.timeout_s) if limit.time else self.timeout_s
        outcome: queue.SimpleQueue = queue.SimpleQueue()

        def work() -> None:
            try:
                infos = self.engine.analyse(board, limit, multipv=self.multipv)
                outcome.put(infos if isinstance(infos, list) else [infos])
            except BaseException as exc:  # noqa: BLE001 - re-raised on caller thread
                outcome.put(exc)

        threading.Thread(target=work, daemon=True).start()
        try:
            result = outcome.get(timeout=self.timeout_s)
        except queue.Empty:
            raise OracleTimeout(f"engine stalled beyond {self.timeout_s}s: {board.fen()}") from None
        if isinstance(result, BaseException):
            raise OracleError(f"engine failed: {result}: {board.fen()}") from result
        return result


class Lc0Oracle(UciOracle):
    def __init__(self, path: str = "lc0", weights: str = "", nodes: int = 64, multipv: int = 4, timeout_s: float = 30.0):
        options = {}
        if weights:
            options["WeightsFile"] = weights
        super().__init__(path, options, multipv, timeout_s)
        self.nodes = nodes

    def analyse(self, board: chess.Board) -> OracleResult:
        return self._analyse(board, chess.engine.Limit(nodes=self.nodes))


class StockfishOracle(UciOracle):
    def __init__(self, path: str = "stockfish", depth: int = 22, multipv: int = 4, timeout_s: float = 30.0):
        super().__init__(path, {}, multipv, timeout_s)
        self.depth = depth

    def analyse(self, board: chess.Board) -> OracleResult:
        return self._analyse(board, chess.engine.Limit(depth=self.depth))
