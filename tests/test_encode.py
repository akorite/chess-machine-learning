import chess
import numpy as np

from chessfoundation.data import build_cache, chessbench
from chessfoundation.data.encode import (
    CP_MAX,
    CP_UNKNOWN,
    PLANES,
    POLICY_ACTIONS,
    board_planes,
    canonical_index,
    canonical_move,
    canonical_planes,
    index_to_move,
    legal_masks,
    move_index,
    promo_index,
    record_from_board,
    record_from_eval,
    set_best_move,
    wdl_from_cp,
    win_prob,
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


def mirror_board(board):
    mirrored = board.copy(stack=False)
    mirrored.apply_mirror()
    return mirrored


def test_canonical_index_is_an_involution():
    indices = np.arange(POLICY_ACTIONS)
    assert np.array_equal(canonical_index(canonical_index(indices)), indices)
    assert canonical_index(move_index(chess.Move.from_uci("e2e4"))) == move_index(
        chess.Move.from_uci("e7e5")
    )


def test_canonical_move_mirrors_and_keeps_promotion():
    move = canonical_move(chess.Move.from_uci("a7b8q"))
    assert move == chess.Move.from_uci("a2b1q")
    assert canonical_move(move) == chess.Move.from_uci("a7b8q")


def test_canonical_planes_are_identical_for_a_position_and_its_mirror():
    for fen in (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b Kq - 4 7",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
        "8/8/4k3/8/8/3K4/8/7R b - - 3 40",
    ):
        board = chess.Board(fen)
        mirrored = mirror_board(board)
        assert np.array_equal(
            canonical_planes(board_planes(board), board.turn == chess.BLACK),
            canonical_planes(board_planes(mirrored), mirrored.turn == chess.BLACK),
        )


def test_canonical_planes_twice_is_identity():
    planes = board_planes(chess.Board())
    assert np.array_equal(canonical_planes(canonical_planes(planes, True), True), planes)


def test_canonical_planes_batch_matches_single_rows():
    boards = [chess.Board(), mirror_board(chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2"))]
    flags = np.array([board.turn == chess.BLACK for board in boards])
    batch = canonical_planes(np.stack([board_planes(board) for board in boards]), flags)
    for row, board in enumerate(boards):
        assert np.array_equal(batch[row], canonical_planes(board_planes(board), flags[row]))


def test_training_batch_views_black_positions_from_the_mover():
    from chessfoundation.train.loop import batch_from_records

    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    record = record_from_board(board)
    set_best_move(record, chess.Move.from_uci("e7e5"))
    batch = batch_from_records([record])
    assert np.array_equal(
        batch["planes"][0].numpy().astype(np.uint8), canonical_planes(board_planes(board), True)
    )
    assert int(batch["policy_idx"][0, 0]) == move_index(chess.Move.from_uci("e2e4"))


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


def test_non_promotion_record_marks_promo_unknown():
    record = record_from_eval(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        cp=0,
        mate=None,
        pv_uci="e2e4",
    )
    assert record["promo"] == 255


def test_promotion_record_marks_piece():
    record = record_from_eval("8/P7/8/8/8/8/8/K6k w - - 0 1", cp=0, mate=None, pv_uci="a7a8q")
    assert record["promo"] == 0


def test_pv_stops_after_six_plies():
    record = record_from_eval(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        cp=0,
        mate=None,
        pv_uci="e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5c6 d7c6",
    )
    assert record["pv"][5] == chess.A7 * 64 + chess.A6


def test_win_prob_is_even_at_dead_equal():
    assert win_prob(0, white_to_move=True) == 0.5
    assert win_prob(0, white_to_move=False) == 0.5


def test_win_prob_from_the_movers_perspective():
    assert win_prob(400, white_to_move=True) > 0.9
    assert win_prob(400, white_to_move=False) < 0.1
    assert win_prob(-400, white_to_move=False) > 0.9


def test_win_prob_saturates_for_mate():
    assert win_prob(CP_MAX, white_to_move=True) == 1.0
    assert win_prob(CP_MAX, white_to_move=False) == 0.0
    assert win_prob(-CP_MAX, white_to_move=False) == 1.0


def test_win_prob_blur_pulls_toward_even():
    assert win_prob(800, white_to_move=True, blurred=400) < win_prob(800, white_to_move=True)


def test_record_from_board_marks_cp_unknown():
    assert record_from_board(chess.Board())["cp"] == CP_UNKNOWN


def test_record_from_eval_stores_white_relative_cp():
    white = record_from_eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", cp=120, mate=None, pv_uci="")
    black = record_from_eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", cp=120, mate=None, pv_uci="")
    assert white["cp"] == 120
    assert black["cp"] == 120
    assert win_prob(black["cp"], white_to_move=False) < 0.5


def test_record_from_eval_stores_clamped_mate_cp():
    win = record_from_eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", cp=None, mate=3, pv_uci="")
    loss = record_from_eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", cp=None, mate=-3, pv_uci="")
    assert win["cp"] == CP_MAX
    assert loss["cp"] == -CP_MAX


def test_chessbench_record_converts_action_value_to_cp(monkeypatch):
    rows = [{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", "move": "e7e5", "value": 0.5}]
    monkeypatch.setattr(chessbench, "iter_actions", lambda *args, **kwargs: iter(rows))
    record = next(build_cache.chessbench_records("unused"))
    # Action value is from the mover's side; stored cp is white-relative.
    assert record["cp"] == -200
    assert win_prob(record["cp"], white_to_move=False) > 0.5


def test_win_prob_rejects_unknown_cp():
    try:
        win_prob(CP_UNKNOWN, white_to_move=True)
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown cp must not produce a silent target")
