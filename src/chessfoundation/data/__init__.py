"""Dataset access. Every loader streams; no source is materialised in memory."""

from .encode import (  # noqa: F401
    MOVES_LEFT_UNKNOWN,
    PLANES,
    POLICY_ACTIONS,
    RECORD_DTYPE,
    WDL_UNKNOWN,
    board_planes,
    index_to_move,
    legal_masks,
    move_index,
    planes_from_bits,
    promo_index,
    record_from_board,
    record_from_eval,
)
