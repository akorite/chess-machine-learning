"""Teacher oracles: Lc0 and Stockfish through python-chess's UCI client."""

from __future__ import annotations

import math
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


def value_from_score(score: chess.engine.PovScore, turn: chess.Color) -> float:
    pov = score.pov(turn)
    if pov.is_mate():
        mate = pov.mate()
        return 1.0 if mate and mate > 0 else -1.0
    cp = pov.score(mate_score=100_000)
    if cp is None:
        return 0.0
    return 2.0 / (1.0 + math.exp(-cp / 300.0)) - 1.0


def policy_from_scores(infos: list[chess.engine.InfoDict], temperature: float = 120.0) -> dict[chess.Move, float]:
    scores: dict[chess.Move, float] = {}
    for info in infos:
        pv = info.get("pv") or ()
        score = info.get("score")
        if not pv or score is None:
            continue
        cp = score.white().score(mate_score=100_000) or 0
        scores[pv[0]] = math.exp(cp / temperature)
    total = sum(scores.values())
    if total <= 0:
        return {}
    return {move: weight / total for move, weight in scores.items()}


class UciOracle:
    """Base for engines that answer through UCI analyse."""

    def __init__(self, path: str, options: dict | None = None, multipv: int = 4):
        self.path = path
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

    def analyse(self, board: chess.Board) -> OracleResult:
        raise NotImplementedError


class Lc0Oracle(UciOracle):
    def __init__(self, path: str = "lc0", weights: str = "", nodes: int = 64, multipv: int = 4):
        options = {"MultiPV": multipv}
        if weights:
            options["WeightsFile"] = weights
        super().__init__(path, options, multipv)
        self.nodes = nodes

    def analyse(self, board: chess.Board) -> OracleResult:
        assert self.engine is not None, "oracle not open"
        infos = self.engine.analyse(board, chess.engine.Limit(nodes=self.nodes), multipv=self.multipv)
        infos = infos if isinstance(infos, list) else [infos]
        policy = policy_from_scores(infos)
        best = infos[0]
        pv = tuple(best.get("pv") or ())
        value = value_from_score(best["score"], board.turn) if best.get("score") else 0.0
        return OracleResult(
            move=pv[0] if pv else next(iter(board.legal_moves)),
            value=value,
            pv=pv,
            policy=policy,
            nodes=int(best.get("nodes") or self.nodes),
        )


class StockfishOracle(UciOracle):
    def __init__(self, path: str = "stockfish", depth: int = 22, multipv: int = 4):
        super().__init__(path, {"MultiPV": multipv}, multipv)
        self.depth = depth

    def analyse(self, board: chess.Board) -> OracleResult:
        assert self.engine is not None, "oracle not open"
        infos = self.engine.analyse(board, chess.engine.Limit(depth=self.depth), multipv=self.multipv)
        infos = infos if isinstance(infos, list) else [infos]
        policy = policy_from_scores(infos)
        best = infos[0]
        pv = tuple(best.get("pv") or ())
        value = value_from_score(best["score"], board.turn) if best.get("score") else 0.0
        return OracleResult(
            move=pv[0] if pv else next(iter(board.legal_moves)),
            value=value,
            pv=pv,
            policy=policy,
            nodes=int(best.get("nodes") or 0),
        )
