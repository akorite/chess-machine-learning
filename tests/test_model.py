from unittest import mock

import chess
import numpy as np
import torch

from chessfoundation.config import PRESETS
from chessfoundation.data.encode import PLANES, POLICY_ACTIONS, board_planes
from chessfoundation.model.presets import TARGETS, build_student
from chessfoundation.model.student import parameter_count


def position_planes(*boards: chess.Board) -> torch.Tensor:
    return torch.from_numpy(np.stack([board_planes(board) for board in boards])).float()


def attention_biases(model, planes: torch.Tensor) -> torch.Tensor:
    """Stack the per-layer bias tensors the attention layers hand to SDPA."""
    biases: list[torch.Tensor] = []
    original = torch.nn.functional.scaled_dot_product_attention

    def spy(q, k, v, attn_mask=None, **kwargs):
        biases.append(attn_mask.detach().clone())
        return original(q, k, v, attn_mask=attn_mask, **kwargs)

    with (
        mock.patch.object(torch.nn.functional, "scaled_dot_product_attention", spy),
        torch.no_grad(),
    ):
        model(planes)
    return torch.stack(biases)


def block_stack_output(model, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        for block in model.blocks:
            tokens = block(tokens)
        return model.heads(tokens)


def test_forward_shapes_and_normalisation():
    model = build_student("p6m").eval()
    with torch.no_grad():
        outputs = model(torch.zeros(2, PLANES, 8, 8))
    assert outputs["policy"].shape == (2, POLICY_ACTIONS)
    assert outputs["promo"].shape == (2, 4)
    assert outputs["wdl"].shape == (2, 3)
    assert outputs["pv"].shape == (2, 6, POLICY_ACTIONS)
    assert outputs["moves_left"].shape == (2,)
    assert outputs["value"].shape == (2,)
    assert torch.allclose(outputs["wdl"].softmax(-1).sum(-1), torch.ones(2))
    assert outputs["wdl"].shape == (2, 3)
    assert outputs["value"].abs().le(1).all()
    assert torch.isfinite(outputs["moves_left"]).all()


def test_pv_and_policy_projections_are_independent():
    torch.manual_seed(0)
    model = build_student("p6m").eval()
    planes = position_planes(chess.Board(), chess.Board())
    with torch.no_grad():
        before = model(planes)
        model.heads.src.weight.normal_(std=0.05)
        policy_moved = model(planes)
        model.heads.pv_src.weight.normal_(std=0.05)
        pv_moved = model(planes)

    assert torch.allclose(before["pv"], policy_moved["pv"])
    assert not torch.allclose(before["policy"], policy_moved["policy"])
    assert torch.allclose(policy_moved["policy"], pv_moved["policy"])
    assert not torch.allclose(policy_moved["pv"], pv_moved["pv"])


def test_presets_sit_within_parameter_budget():
    for name in ("p6m", "p20m"):
        model = build_student(name)
        count = parameter_count(model)
        assert abs(count - TARGETS[name]) <= 0.05 * TARGETS[name], (name, count)


def test_pure_transformer_variant_instantiates():
    model = build_student("p6m", pure_transformer=True)
    with torch.no_grad():
        outputs = model(torch.zeros(1, PLANES, 8, 8))
    assert outputs["policy"].shape == (1, POLICY_ACTIONS)
    assert model.stem is None


def test_policy_logits_are_per_source_destination_pair():
    model = build_student("p6m").eval()
    with torch.no_grad():
        outputs = model(torch.zeros(1, PLANES, 8, 8))
    grid = outputs["policy"].reshape(64, 64)
    assert grid.shape == (64, 64)
    assert torch.isfinite(grid).all()


def test_preset_registry_matches_config():
    assert PRESETS["p6m"].pv_plies == 6


def test_permuting_square_contents_changes_the_output():
    torch.manual_seed(0)
    model = build_student("p6m").eval()
    planes = position_planes(chess.Board())

    perm = torch.randperm(64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        content = (
            model.embed(planes) - model.pos_embed
        )  # what a square holds, minus where
        permuted = block_stack_output(model, content[:, perm] + model.pos_embed)[
            "policy"
        ]
        reference = model(planes)["policy"]

    # Sorted logits stay insensitive to ordering, so a pass cannot come from reindexing alone.
    assert not torch.allclose(
        reference.flatten().sort().values, permuted.flatten().sort().values, atol=1e-2
    )


def test_attention_bias_keeps_per_square_structure():
    torch.manual_seed(0)
    model = build_student("p6m").eval()
    bias = attention_biases(model, torch.zeros(1, PLANES, 8, 8))[0, 0, 0]
    singular = torch.linalg.svdvals(bias)
    # A bias generated from one board-wide vector can only ever be rank 1.
    assert (singular > 1e-3 * singular[0]).sum() > 1


def test_attention_bias_depends_on_piece_placement():
    torch.manual_seed(0)
    model = build_student("p6m").eval()
    current = chess.Board()
    shifted = chess.Board()
    shifted.set_piece_at(chess.E4, shifted.remove_piece_at(chess.E2), promoted=False)

    first = attention_biases(model, position_planes(current))
    second = attention_biases(model, position_planes(shifted))

    assert first.shape[-2:] == (64, 64)
    assert not torch.allclose(first, second)


def test_output_is_invariant_to_batch_composition():
    torch.manual_seed(0)
    model = build_student("p6m").eval()
    planes = torch.zeros(2, PLANES, 8, 8)
    planes[1, 0, 0, 0] = 1.0
    with torch.no_grad():
        batched = model(planes)
        single = model(planes[1:])
    for key in ("policy", "wdl", "moves_left", "promo"):
        assert torch.allclose(batched[key][1:], single[key], atol=1e-3), key


