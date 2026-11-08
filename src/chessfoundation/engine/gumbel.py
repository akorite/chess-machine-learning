"""Gumbel root selection with a compact MCTS and network leaf values."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import chess
import numpy as np
import torch
from torch import nn

from ..data.encode import (
    board_planes,
    legal_masks,
    move_index,
    promo_index,
)


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 512
    max_considered: int = 16
    c_visit: float = 50.0
    bf16: bool = False
    mode: Literal["gumbel", "puct"] = "gumbel"
    # Wall-clock ceiling in milliseconds; None means sim-count only.
    budget_ms: float | None = None
    # Leaves evaluated per forward pass; 1 restores per-sim evaluation.
    eval_batch: int = 16


DEFAULT_SEARCH = SearchConfig()


@dataclass(frozen=True)
class SearchResult:
    move: chess.Move
    simulations: int


class _Node:
    __slots__ = ("board", "children", "legal", "prior", "promo", "value_sum", "visits", "wdl")

    def __init__(self, board: chess.Board):
        self.board = board
        self.prior: np.ndarray | None = None
        self.promo: np.ndarray | None = None
        self.wdl: np.ndarray | None = None
        self.children: dict[int, _Node] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.legal: list[int] = []

    @property
    def mean(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def _evaluate(model: nn.Module, boards: list[chess.Board], device: str, bf16: bool = False):
    planes = torch.from_numpy(
        np.stack([board_planes(board) for board in boards]).astype(np.float32)
    ).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16 and device == "cuda"):
        outputs = model(planes)
    masks = np.stack([legal_masks(board)[0] for board in boards])
    logits = outputs["policy"].detach().cpu().numpy().astype(np.float64)
    logits = np.where(masks, logits, -np.inf)
    wdl = outputs["wdl"].softmax(-1).detach().cpu().numpy()
    value = (wdl * np.array([-1.0, 0.0, 1.0])).sum(-1)
    promo = outputs["promo"].detach().cpu().numpy()
    return logits, wdl, value, promo


def _terminal_value(board: chess.Board) -> float:
    if board.is_checkmate():
        return -1.0
    if board.is_game_over():
        return 0.0
    return float("nan")


def _expand(model: nn.Module, node: _Node, device: str, bf16: bool = False) -> float:
    logits, wdl, value, promo = _evaluate(model, [node.board], device, bf16)
    node.prior = logits[0]
    node.wdl = wdl[0]
    node.promo = promo[0]
    node.legal = [move_index(move) for move in node.board.legal_moves]
    terminal = _terminal_value(node.board)
    return float(value[0]) if np.isnan(terminal) else terminal


def _select(node: _Node, c_visit: float) -> int | None:
    best_index, best_score = None, -np.inf
    for index in node.legal:
        child = node.children.get(index)
        if child is not None and child.visits == 0:
            continue  # queued for expansion in this batch
        score = node.prior[index] + (c_visit * child.mean if child is not None else 0.0)
        if score > best_score:
            best_index, best_score = index, score
    return best_index


def entropy(wdl: np.ndarray | None) -> float:
    if wdl is None:
        return 0.5
    probs = np.clip(wdl, 1e-9, 1.0)
    return float(-(probs * np.log(probs)).sum() / np.log(3.0))


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
    root = _Node(board.copy(stack=False))
    root_value = _expand(model, root, device, cfg.bf16)
    root.visits = 1
    root.value_sum = root_value

    # More simulations when the result is uncertain.
    budget = max(1, int(cfg.simulations * (0.5 + 0.5 * entropy(root.wdl))))
    if len(legal) == 1 or budget <= 1:
        return SearchResult(_best_move(root, legal), 1)

    gumbel = {}
    if cfg.mode == "gumbel":
        considered = sorted(root.legal, key=lambda i: root.prior[i], reverse=True)[: cfg.max_considered]
        gumbel = {i: -np.log(-np.log(rng.random() + 1e-12) + 1e-12) for i in considered}
    evaluated = 1
    pending: list[tuple[_Node, list[_Node]]] = []
    started = time.monotonic()
    while evaluated + len(pending) < budget:
        if cfg.budget_ms is not None and (time.monotonic() - started) * 1000.0 >= cfg.budget_ms:
            break
        for _ in range(max(1, cfg.eval_batch) - len(pending)):
            if evaluated + len(pending) >= budget:
                break
            if cfg.budget_ms is not None and (time.monotonic() - started) * 1000.0 >= cfg.budget_ms:
                break
            node = root
            path: list[_Node] = []
            value = None
            queued = False
            while True:
                if node.board.is_game_over():
                    value = _terminal_value(node.board)
                    break
                if node is not root and node.visits == 0:
                    break  # already queued in pending
                index = _select_root(root, gumbel, cfg.c_visit) if node is root else _select(node, cfg.c_visit)
                if index is None:
                    break  # every candidate child is queued in pending
                child = node.children.get(index)
                if child is None:
                    child_board = node.board.copy(stack=False)
                    child_board.push(_move_from_index(node, index))
                    child = _Node(child_board)
                    node.children[index] = child
                    path.append(child)
                    if child_board.is_game_over():
                        value = _terminal_value(child_board)
                    else:
                        pending.append((child, path))
                        queued = True
                    break
                node = child
                path.append(node)
            if value is None:
                if not queued:
                    break
                continue
            for item in reversed(path):
                item.visits += 1
                item.value_sum += -value
                value = -value
            root.visits += 1
            evaluated += 1
        if not pending:
            break
        boards = [node.board for node, _ in pending]
        logits, wdl, values, promos = _evaluate(model, boards, device, cfg.bf16)
        for k, (node, path) in enumerate(pending):
            node.prior = logits[k]
            node.wdl = wdl[k]
            node.promo = promos[k]
            node.legal = [move_index(move) for move in node.board.legal_moves]
            value = float(values[k])
            for item in reversed(path):
                item.visits += 1
                item.value_sum += -value
                value = -value
            root.visits += 1
            evaluated += 1
        pending.clear()

    return SearchResult(_best_move(root, legal, gumbel, cfg.c_visit, cfg.mode), evaluated)


def _select_root(root: _Node, gumbel: dict[int, float], c_visit: float) -> int | None:
    best_index, best_score = None, -np.inf
    for index in root.legal:
        if gumbel and index not in gumbel:
            continue
        child = root.children.get(index)
        if child is not None and child.visits == 0:
            continue  # queued for expansion in this batch
        score = root.prior[index] + gumbel.get(index, 0.0) + (
            c_visit * child.mean if child is not None else 0.0
        )
        if score > best_score:
            best_index, best_score = index, score
    return best_index

def _move_from_index(node: _Node, index: int) -> chess.Move:
    from_square, to_square = divmod(index, 64)
    candidates = [
        move
        for move in node.board.legal_moves
        if move.from_square == from_square and move.to_square == to_square
    ]
    quiet = [move for move in candidates if move.promotion is None]
    if quiet:
        return quiet[0]
    if candidates:
        if node.promo is not None:
            return max(candidates, key=lambda move: node.promo[promo_index(move)])
        return candidates[0]
    raise ValueError(f"index {index} is not legal in {node.board.fen()}")


def _promotion_choice(root: _Node, from_square: int, to_square: int) -> chess.Move | None:
    if root.promo is None:
        return None
    candidates = [
        move
        for move in root.board.legal_moves
        if move.from_square == from_square and move.to_square == to_square
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda move: root.promo[promo_index(move)])


def _best_move(
    root: _Node,
    legal: list[chess.Move],
    gumbel: dict[int, float] | None = None,
    c_visit: float = 0.0,
    mode: str = "gumbel",
) -> chess.Move:
    def score(move: chess.Move):
        index = move_index(move)
        child = root.children.get(index)
        if mode == "puct":
            visits = float(child.visits) if child is not None else 0.0
            return (visits, root.prior[index])
        base = root.prior[index] + (gumbel[index] if gumbel and index in gumbel else 0.0)
        return base + (c_visit * child.mean if child is not None else 0.0)

    considered_moves = [m for m in legal if not gumbel or move_index(m) in gumbel] or legal
    best = max(considered_moves, key=score)
    if best.promotion is not None:
        choice = _promotion_choice(root, best.from_square, best.to_square)
        if choice is not None:
            return choice
    return best
