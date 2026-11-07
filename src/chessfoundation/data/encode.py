"""Board encoding shared by training, cache, and inference."""

from __future__ import annotations

import chess
import numpy as np

PLANES = 20
POLICY_ACTIONS = 64 * 64
PROMO_PIECES = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
PIECE_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
WDL_UNKNOWN = 255
MOVES_LEFT_UNKNOWN = 255
DRAW_CP = 50

RECORD_DTYPE = np.dtype(
    [
        ("pieces", "<u8", (12,)),
        ("meta", "u1", (5,)),
        ("policy", "<u2", (8,)),
        ("policy_prob", "<u2", (8,)),
        ("promo", "u1"),
        ("wdl", "u1"),
        ("moves_left", "u1"),
        ("pv", "<u2", (6,)),
    ]
)


def move_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def index_to_move(index: int) -> tuple[int, int]:
    return divmod(int(index), 64)


def promo_index(move: chess.Move) -> int:
    if move.promotion is None:
        return -1
    return PROMO_PIECES.index(move.promotion)


def promo_move(from_square: int, to_square: int, promo: int) -> chess.Move:
    return chess.Move(from_square, to_square, PROMO_PIECES[promo])


def _bits(board: chess.Board) -> np.ndarray:
    bits = np.zeros(12, dtype="<u8")
    for color_idx, color in enumerate((chess.WHITE, chess.BLACK)):
        for piece_idx, piece_type in enumerate(PIECE_ORDER):
            bits[color_idx * 6 + piece_idx] = board.pieces_mask(piece_type, color)
    return bits


def _meta(board: chess.Board) -> np.ndarray:
    castling = (
        board.has_kingside_castling_rights(chess.WHITE)
        | board.has_queenside_castling_rights(chess.WHITE) << 1
        | board.has_kingside_castling_rights(chess.BLACK) << 2
        | board.has_queenside_castling_rights(chess.BLACK) << 3
    )
    ep = board.ep_square + 1 if board.ep_square is not None else 0
    return np.array(
        [castling, ep, int(board.turn == chess.WHITE), min(board.halfmove_clock, 255),
         int(board.is_check())],
        dtype="u1",
    )


def planes_from_bits(bits: np.ndarray, meta: np.ndarray) -> np.ndarray:
    raw = np.asarray(bits, dtype="<u8").tobytes()
    bytes_2d = np.frombuffer(raw, dtype=np.uint8).reshape(12, 8)
    planes = np.unpackbits(bytes_2d, axis=1, bitorder="little").reshape(12, 8, 8)
    planes = planes.astype(np.uint8)

    out = np.zeros((PLANES, 8, 8), dtype=np.uint8)
    out[:12] = planes
    for i in range(4):
        out[12 + i] = (int(meta[0]) >> i) & 1
    if meta[1]:
        square = int(meta[1]) - 1
        out[16, square // 8, square % 8] = 1
    out[17] = int(meta[2]) == 1
    out[18] = min(int(meta[3]), 100)
    out[19] = int(meta[4])
    return out


def board_planes(board: chess.Board) -> np.ndarray:
    return planes_from_bits(_bits(board), _meta(board))


def record_from_board(board: chess.Board) -> np.ndarray:
    record = np.zeros((), dtype=RECORD_DTYPE)
    record["pieces"] = _bits(board)
    record["meta"] = _meta(board)
    record["promo"] = 255
    record["wdl"] = WDL_UNKNOWN
    record["moves_left"] = MOVES_LEFT_UNKNOWN
    return record


def legal_masks(board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
    policy = np.zeros(POLICY_ACTIONS, dtype=bool)
    promo = np.zeros(4, dtype=bool)
    for move in board.legal_moves:
        policy[move_index(move)] = True
        index = promo_index(move)
        if index >= 0:
            promo[index] = True
    return policy, promo


def wdl_from_cp(cp: int, white_to_move: bool, threshold: int = DRAW_CP) -> int:
    stm = cp if white_to_move else -cp
    if stm > threshold:
        return 2
    if stm < -threshold:
        return 0
    return 1


def wdl_from_mate(mate: int, white_to_move: bool) -> int:
    return 2 if (mate > 0) == white_to_move else 0


def record_from_eval(
    fen: str,
    cp: int | None,
    mate: int | None,
    pv_uci: str,
    moves_left: int = MOVES_LEFT_UNKNOWN,
) -> np.ndarray:
    board = chess.Board(fen)
    record = record_from_board(board)
    white_to_move = board.turn == chess.WHITE
    if mate is not None:
        record["wdl"] = wdl_from_mate(mate, white_to_move)
    elif cp is not None:
        record["wdl"] = wdl_from_cp(cp, white_to_move)
    unknown = moves_left is None or int(moves_left) >= MOVES_LEFT_UNKNOWN
    record["moves_left"] = MOVES_LEFT_UNKNOWN if unknown else int(moves_left)

    moves: list[chess.Move] = []
    for token in pv_uci.split():
        try:
            move = chess.Move.from_uci(token)
        except ValueError:
            break
        if not board.is_legal(move):
            break
        moves.append(move)
        board.push(move)
    if moves:
        record["policy"][0] = move_index(moves[0])
        record["policy_prob"][0] = 65535
        record["promo"] = max(promo_index(moves[0]), 0)
        for i, move in enumerate(moves[:6]):
            record["pv"][i] = move_index(move)
    return record
