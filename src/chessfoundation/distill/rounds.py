"""On-policy distillation rounds: play, label, retrain, gate."""

from __future__ import annotations

import random
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import chess
import chess.engine
import numpy as np

from ..config import TrainConfig
from ..data.cache import ShardDataset, write_manifest, write_shard
from ..data.encode import move_index, promo_index, record_from_board
from ..engine.uci import EngineSpec
from ..eval.tournament import MatchConfig, MatchResult, play_game, run_local
from ..teacher.avi import wdl_from_value
from ..train import loop
from .replay import ConcatDataset, reanalyse


def _collect_positions(board: chess.Board, start_ply: int, max_plies: int) -> list[chess.Board]:
    """Replay the finished game's moves, snapshotting each pre-move position."""
    walker = board.copy(stack=True)
    while len(walker.move_stack) > start_ply:
        walker.pop()
    positions = []
    for move in list(board.move_stack)[start_ply:][:max_plies]:
        if walker.is_game_over(claim_draw=True):
            break
        positions.append(walker.copy(stack=False))
        walker.push(move)
    return positions


def play_and_collect(
    white: chess.engine.SimpleEngine,
    black: chess.engine.SimpleEngine,
    games: int = 2,
    movetime_ms: int = 20,
    max_plies: int = 200,
    openings: tuple[str, ...] = (),
    rng: random.Random | None = None,
) -> list[chess.Board]:
    rng = rng or random.Random(0)
    engines = {chess.WHITE: white, chess.BLACK: black}
    limits = {engine: chess.engine.Limit(time=movetime_ms / 1000.0) for engine in engines.values()}
    positions: list[chess.Board] = []
    for _ in range(games):
        board = chess.Board()
        if openings:
            for token in rng.choice(openings).split():
                board.push_uci(token)
        start_ply = board.ply()
        play_game(engines, board, limits)  # fault-isolated: a dead engine forfeits this game
        positions.extend(_collect_positions(board, start_ply, max_plies))
    return positions


def label_record(board: chess.Board, value: float, policy: dict[chess.Move, float]) -> np.ndarray:
    record = record_from_board(board)
    record["wdl"] = wdl_from_value(value)
    if policy:
        ranked = sorted(policy.items(), key=lambda item: item[1], reverse=True)[:8]
        total = sum(weight for _, weight in ranked) or 1.0
        for i, (move, weight) in enumerate(ranked):
            if not board.is_legal(move):
                continue
            record["policy"][i] = move_index(move)
            record["policy_prob"][i] = min(65535, int(65535 * weight / total))
        if record["policy_prob"][0]:
            first_legal = next((m for m, _ in ranked if board.is_legal(m)), None)
            if first_legal is not None:
                index = promo_index(first_legal)
                record["promo"] = index if index >= 0 else 255
    return record


def label_positions(positions: list[chess.Board], analyse: Callable[[chess.Board], tuple[float, dict[chess.Move, float]]]) -> list[np.ndarray]:
    from ..teacher.lc0_oracle import OracleError

    records = []
    skipped = 0
    for board in positions:
        try:
            value, policy = analyse(board)
        except OracleError as exc:
            skipped += 1
            print(f"warning: label skipped: {exc}", file=sys.stderr)
            continue
        records.append(label_record(board, value, policy))
    if skipped:
        print(f"warning: labeling skipped {skipped}/{len(positions)} positions (oracle faults)", file=sys.stderr)
    return records


def run_round(
    model,
    positions: list[chess.Board],
    analyse: Callable[[chess.Board], tuple[float, dict[chess.Move, float]]],
    cfg: TrainConfig,
    round_dir: str | Path,
    replay_dirs: tuple[str, ...] = (),
    device: str = "cpu",
    reanalyse_sample: int = 0,
) -> dict:
    records = label_positions(positions, analyse)
    if not records:
        raise ValueError("round produced no positions")
    round_dir = Path(round_dir)
    round_dir.mkdir(parents=True, exist_ok=True)
    write_shard(round_dir / "shard-00000.bin", np.stack(records))
    write_manifest(round_dir, {"records": len(records)})
    parts: list = [records]
    fresh = []
    if replay_dirs:
        replay = ConcatDataset([ShardDataset(d) for d in replay_dirs if Path(d).exists()])
        if reanalyse_sample and len(replay):
            fresh = reanalyse(
                replay, lambda board: label_record(board, *analyse(board)), reanalyse_sample, cfg.seed
            )
            parts.insert(0, fresh)
        elif len(replay):
            parts.insert(0, replay)
    dataset = ConcatDataset(parts)
    metrics = loop.train(model, dataset, cfg, cfg.max_steps, device)
    metrics["records"] = len(records)
    metrics["reanalysed"] = len(fresh)
    return metrics


def gate_match(
    student: EngineSpec,
    rung: EngineSpec,
    games: int,
    movetime_ms: int,
    nodes: int | None = None,
    openings: tuple[str, ...] = (),
    pgn_path: str = "",
    seed: int = 0,
) -> float:
    """Match score of the round checkpoint vs the rung; rung takes the `black` spec."""
    result: MatchResult = run_local(
        MatchConfig(
            white=student,
            black=rung,
            games=games,
            movetime_ms=movetime_ms,
            nodes=nodes,
            openings=openings,
            pgn_path=pgn_path,
        ),
        seed=seed,
    )
    return result.score


def apply_gate(
    out_dir: Path,
    round_index: int,
    round_ckpt: Path,
    score: float | None,
    history: list[dict],
    gate_error: str | None = None,
) -> dict:
    """Accept the round when its match score holds or improves vs the last accepted round.

    An unmeasured score (no gate configured) accepts; a failed gate match is a
    rejection, never a silent promotion.
    """
    previous = next((r["score"] for r in reversed(history) if r.get("accepted")), None)
    accepted = gate_error is None and (score is None or previous is None or score >= previous)
    if accepted:
        shutil.copyfile(round_ckpt, out_dir / "current.pt")
    else:
        round_ckpt.unlink(missing_ok=True)
    return {
        "round": round_index,
        "score": score,
        "previous_score": previous,
        "accepted": accepted,
        "gate_error": gate_error,
        "dir": f"round-{round_index:04d}",
    }
