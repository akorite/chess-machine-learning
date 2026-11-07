"""Gumbel search with sequential halving over one-step leaf values."""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np
import torch
from torch import nn

from ..data.encode import board_planes, index_to_move, legal_masks, move_index


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 512
    max_considered: int = 16
    c_scale: float = 1.0


DEFAULT_SEARCH = SearchConfig()


@dataclass(frozen=True)
class SearchResult:
    move: chess.Move
    simulations: int


def value_of(outputs: dict[str, torch.Tensor]) -> np.ndarray:
    probs = outputs["wdl"].softmax(-1)
    weights = torch.tensor([-1.0, 0.0, 1.0], device=probs.device)
    return (probs * weights).sum(-1).detach().cpu().numpy()


def masked_policy(outputs: dict[str, torch.Tensor], masks: np.ndarray) -> np.ndarray:
    logits = outputs["policy"].detach().cpu().numpy().astype(np.float64)
    logits = np.where(masks, logits, -np.inf)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted) * masks
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def evaluate(model: nn.Module, boards: list[chess.Board], device: str) -> tuple[np.ndarray, np.ndarray]:
    planes = torch.from_numpy(
        np.stack([board_planes(board) for board in boards]).astype(np.float32)
    ).to(device)
    with torch.no_grad():
        outputs = model(planes)
    masks = np.stack([legal_masks(board)[0] for board in boards])
    return masked_policy(outputs, masks), value_of(outputs)


def search(
    model: nn.Module,
    board: chess.Board,
    cfg: SearchConfig = DEFAULT_SEARCH,
    rng: np.random.Generator | None = None,
) -> SearchResult:
    rng = rng or np.random.default_rng()
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError("no legal moves")
    device = next(model.parameters()).device.type
    policy, _ = evaluate(model, [board], device)
    if len(legal) == 1:
        return SearchResult(legal[0], 1)

    scores = np.array([policy[0, move_index(move)] for move in legal], dtype=np.float64)
    if cfg.simulations <= 1:
        best = int(np.argmax(scores))
        return SearchResult(legal[best], 1)

    gumbel = -np.log(-np.log(np.clip(rng.random(len(legal)), 1e-9, 1 - 1e-9)))
    order = np.argsort(scores + gumbel)[::-1]
    considered = list(order[: min(len(legal), cfg.max_considered, cfg.simulations)])
    q: dict[int, float] = {}
    evaluated = 0
    while len(considered) > 1:
        remaining = cfg.simulations - evaluated
        if remaining <= 0:
            break
        pending = [index for index in considered if index not in q][:remaining]
        if pending:
            children = []
            for index in pending:
                child = board.copy(stack=False)
                child.push(legal[index])
                children.append(child)
            _, values = evaluate(model, children, device)
            for index, child, value in zip(pending, children, values):
                q[index] = 1.0 if child.is_checkmate() else -float(value)
            evaluated += len(pending)
        final = scores + gumbel + cfg.c_scale * np.array([q.get(i, 0.0) for i in range(len(legal))])
        considered = sorted(considered, key=lambda i: final[i], reverse=True)[: max(1, len(considered) // 2)]
    return SearchResult(legal[considered[0]], evaluated)


def move_from_index(index: int, legal: list[chess.Move]) -> chess.Move:
    from_square, to_square = index_to_move(index)
    for move in legal:
        if move.from_square == from_square and move.to_square == to_square:
            return move
    raise ValueError(f"index {index} is not legal")
