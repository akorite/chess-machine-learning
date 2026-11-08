"""Game results, Elo, intervals, fastchess invocation, and a local runner."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from ..engine.uci import EngineSpec

Z95 = 1.959964
_RESULT = re.compile(r"(\d+)\s+wins?,\s*(\d+)\s+draws?,\s*(\d+)\s+losses?", re.IGNORECASE)


@dataclass(frozen=True)
class MatchConfig:
    white: EngineSpec
    black: EngineSpec
    games: int = 400
    movetime_ms: int = 50
    openings: tuple[str, ...] = ()
    nodes: int | None = None  # per-move node limit pinning the `black` spec (rung opponent)
    pgn_path: str = ""


@dataclass(frozen=True)
class MatchResult:
    wins: int
    draws: int
    losses: int
    pgn_path: str = ""

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    @property
    def elo(self) -> float:
        return elo_from_score(self.score)


def elo_from_score(score: float) -> float:
    if score <= 0.0:
        return -math.inf
    if score >= 1.0:
        return math.inf
    return -400.0 * math.log10(1.0 / score - 1.0)


def score_interval(wins: int, draws: int, losses: int, z: float = Z95) -> tuple[float, float, float]:
    n = wins + draws + losses
    if n == 0:
        raise ValueError("no games")
    score = (wins + 0.5 * draws) / n
    se = math.sqrt(max(score * (1.0 - score), 1e-12) / n)
    return score, max(0.0, score - z * se), min(1.0, score + z * se)


def interval_excludes_even(score: float, low: float, high: float) -> bool:
    return low > 0.5 or high < 0.5


def fastchess_command(cfg: MatchConfig, binary: str = "fastchess") -> list[str]:
    return [
        binary,
        "-engine",
        f"cmd={cfg.white.path}",
        "name=white",
        "tc=inf",
        "-engine",
        f"cmd={cfg.black.path}",
        "name=black",
        "tc=inf",
        "-each",
        "timemargin=0",
        f"st={cfg.movetime_ms / 1000.0:g}",
        "-games",
        str(cfg.games),
        "-rounds",
        "1",
        "-repeat",
        "-pgnout",
        cfg.pgn_path or "match.pgn",
    ]


def parse_fastchess(text: str) -> tuple[int, int, int]:
    match = _RESULT.search(text)
    if not match:
        raise ValueError("no 'N wins, N draws, N losses' line in fastchess output")
    return tuple(int(group) for group in match.groups())  # type: ignore[return-value]


def _info_comment(info: object) -> str:
    """Render an engine info dict as a PGN comment; '' when nothing was reported."""
    if not info or not isinstance(info, dict):
        return ""
    parts = []
    if info.get("nodes") is not None:
        parts.append(f"nodes {info['nodes']}")
    if info.get("depth") is not None:
        parts.append(f"depth {info['depth']}")
    return "info " + " ".join(parts) if parts else ""


def play_game(
    engine_by_colour: dict[chess.Color, chess.engine.SimpleEngine],
    board: chess.Board,
    limits: dict,
    comments: list[str] | None = None,
) -> tuple[chess.Color | None, str | None]:
    """Play one game; engine faults forfeit the side to move, never the match."""
    try:
        while not board.is_game_over(claim_draw=True):
            engine = engine_by_colour[board.turn]
            try:
                result = engine.play(board, limits[engine], info=chess.engine.INFO_ALL)
            except TypeError:
                result = engine.play(board, limits[engine])
            move = result.move
            if move is None:
                return (not board.turn), "stalled"
            board.push(move)
            if comments is not None:
                comments.append(_info_comment(result.info))
    except Exception as exc:  # noqa: BLE001 - a dead engine forfeits this game only
        return (not board.turn), type(exc).__name__
    outcome = board.outcome(claim_draw=True)
    return (outcome.winner if outcome is not None else None), None


def run_local(cfg: MatchConfig, seed: int = 0) -> MatchResult:
    """Smoke-grade runner with a fixed per-move budget.

    The 1+0 measurement uses fastchess; this keeps the harness runnable on a
    machine without the binary. Colours alternate so both engines get both sides.
    """
    rng = random.Random(seed)
    path = Path(cfg.pgn_path or "match.pgn")
    path.parent.mkdir(parents=True, exist_ok=True)
    engines = {chess.WHITE: cfg.white.open(), chess.BLACK: cfg.black.open()}
    limits = {
        engines[chess.WHITE]: chess.engine.Limit(time=cfg.movetime_ms / 1000.0),
        engines[chess.BLACK]: (
            chess.engine.Limit(nodes=cfg.nodes)
            if cfg.nodes is not None
            else chess.engine.Limit(time=cfg.movetime_ms / 1000.0)
        ),
    }
    wins = draws = losses = 0
    games: list[str] = []
    try:
        for game_index in range(cfg.games):
            board = chess.Board()
            if cfg.openings:
                for token in rng.choice(cfg.openings).split():
                    board.push_uci(token)
            spec_colour = chess.WHITE if game_index % 2 == 0 else chess.BLACK
            engine_by_colour = (
                {chess.WHITE: engines[chess.WHITE], chess.BLACK: engines[chess.BLACK]}
                if spec_colour == chess.WHITE
                else {chess.WHITE: engines[chess.BLACK], chess.BLACK: engines[chess.WHITE]}
            )
            comments: list[str] = [""] * board.ply()
            winner, fault = play_game(engine_by_colour, board, limits, comments)
            if winner is None:
                draws += 1
            elif winner == spec_colour:
                wins += 1
            else:
                losses += 1
            game = chess.pgn.Game.from_board(board)
            for node, comment in zip(game.mainline(), comments):
                if comment:
                    node.comment = comment
            game.headers["White"] = "white" if spec_colour == chess.WHITE else "black"
            game.headers["Black"] = "black" if spec_colour == chess.WHITE else "white"
            if fault:
                game.headers["Termination"] = f"fault: {fault}"
            games.append(str(game))
            path.write_text("\n\n".join(games))
    finally:
        for engine in engines.values():
            try:
                engine.quit()
            except chess.engine.EngineError:
                pass
    path = Path(cfg.pgn_path or "match.pgn")
    return MatchResult(wins, draws, losses, str(path))
