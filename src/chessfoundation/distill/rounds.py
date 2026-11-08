"""On-policy distillation rounds: play, label, retrain, gate."""

from __future__ import annotations

import random
import sys
from collections.abc import Callable
from pathlib import Path

import chess
import chess.engine
import numpy as np

from ..config import TrainConfig
from ..data.cache import write_manifest, write_shard
from ..data.encode import move_index, promo_index, record_from_board
from ..teacher.avi import wdl_from_value
from ..train import loop
from .replay import ConcatDataset


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
    positions: list[chess.Board] = []
    for _ in range(games):
        board = chess.Board()
        if openings:
            for token in rng.choice(openings).split():
                board.push_uci(token)
        engines = {chess.WHITE: white, chess.BLACK: black}
        for _ in range(max_plies):
            if board.is_game_over(claim_draw=True):
                break
            positions.append(board.copy(stack=False))
            move = engines[board.turn].play(board, chess.engine.Limit(time=movetime_ms / 1000.0)).move
            if move is None:
                break
            board.push(move)
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
) -> dict:
    records = label_positions(positions, analyse)
    if not records:
        raise ValueError("round produced no positions")
    round_dir = Path(round_dir)
    round_dir.mkdir(parents=True, exist_ok=True)
    write_shard(round_dir / "shard-00000.bin", np.stack(records))
    write_manifest(round_dir, {"records": len(records)})
    parts = [records]
    for directory in replay_dirs:
        if Path(directory).exists():
            from ..data.cache import ShardDataset

            parts.insert(0, ShardDataset(directory))
    dataset = ConcatDataset(parts)
    metrics = loop.train(model, dataset, cfg, cfg.max_steps, device)
    metrics["records"] = len(records)
    return metrics
