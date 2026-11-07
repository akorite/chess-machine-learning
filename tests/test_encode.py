import chess

from chessfoundation.data.encode import (
    PLANES,
    POLICY_ACTIONS,
    board_planes,
    index_to_move,
    legal_masks,
    move_index,
    promo_index,
    record_from_eval,
    wdl_from_cp,
)


def test_start_position_planes_are_exact():
    planes = board_planes(chess.Board())
    assert planes.shape == (PLANES, 8, 8)
    # 16 pieces + 16 pieces filled, side-to-move plane set for white.
    assert planes[:12].sum() == 32
    assert planes[17].all()
    # white pawn on e2: file 4, rank 1
    assert planes[0, 1, 4] == 1
    # black pawn on e7: file 4, rank 6
    assert planes[6, 6, 4] == 1


def test_castling_en_passant_and_check_planes():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    planes = board_planes(board)
    assert planes[12].all() and planes[13].all() and planes[14].all() and planes[15].all()

    ep = chess.Board("rnbqkbnr/ppp1p1pp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
    assert board_planes(ep)[16, 5, 3] == 1


def test_move_index_round_trips_and_promotion():
    move = chess.Move.from_uci("e2e4")
    assert move_index(move) == chess.E2 * 64 + chess.E4
    assert index_to_move(move_index(move)) == (chess.E2, chess.E4)
    assert promo_index(chess.Move.from_uci("e7e8q")) == 0
    assert promo_index(chess.Move.from_uci("e7e8n")) == 3
    assert promo_index(chess.Move.from_uci("e2e4")) == -1


def test_legal_mask_excludes_illegal_targets():
    board = chess.Board()
    policy, promo = legal_masks(board)
    assert policy.shape == (POLICY_ACTIONS,)
    assert policy.reshape(64, 64)[chess.E2, chess.E4]
    assert not policy.reshape(64, 64)[chess.E2, chess.E5]
    assert policy.sum() == 20
    assert not promo.any()


def test_legal_mask_covers_four_promotions():
    board = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
    policy, promo = legal_masks(board)
    assert policy.reshape(64, 64)[chess.A7, chess.A8]
    assert promo.all()


def test_wdl_from_side_to_move_perspective():
    assert wdl_from_cp(300, white_to_move=True) == 2
    assert wdl_from_cp(300, white_to_move=False) == 0
    assert wdl_from_cp(10, white_to_move=True) == 1
    assert wdl_from_cp(-300, white_to_move=False) == 2


def test_record_from_eval_keeps_policy_pv_and_wdl():
    record = record_from_eval(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        cp=120,
        mate=None,
        pv_uci="e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
        moves_left=80,
    )
    assert record["wdl"] == 2
    assert record["policy"][0] == chess.E2 * 64 + chess.E4
    assert record["policy_prob"][0] == 65535
    assert [int(m) for m in record["pv"]][:6] == [
        chess.E2 * 64 + chess.E4,
        chess.E7 * 64 + chess.E5,
        chess.G1 * 64 + chess.F3,
        chess.B8 * 64 + chess.C6,
        chess.F1 * 64 + chess.B5,
        chess.A7 * 64 + chess.A6,
    ]
    assert record["moves_left"] == 80
    assert record["meta"][2] == 1


def test_record_from_eval_flips_score_for_black():
    record = record_from_eval(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
        cp=120,
        mate=None,
        pv_uci="e7e5",
    )
    assert record["wdl"] == 0
    assert record["meta"][2] == 0


def test_record_without_pv_has_no_policy_target():
    record = record_from_eval(
        "8/8/8/8/8/8/8/K6k w - - 0 1", cp=None, mate=None, pv_uci=""
    )
    assert record["policy_prob"][0] == 0
    assert record["moves_left"] == 255
