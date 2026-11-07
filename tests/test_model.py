import torch

from chessfoundation.config import PRESETS
from chessfoundation.data.encode import PLANES, POLICY_ACTIONS
from chessfoundation.model.presets import TARGETS, build_student
from chessfoundation.model.student import parameter_count


def test_forward_shapes_and_normalisation():
    model = build_student("p6m").eval()
    with torch.no_grad():
        outputs = model(torch.zeros(2, PLANES, 8, 8))
    assert outputs["policy"].shape == (2, POLICY_ACTIONS)
    assert outputs["promo"].shape == (2, 4)
    assert outputs["wdl"].shape == (2, 3)
    assert outputs["pv"].shape == (2, 6, POLICY_ACTIONS)
    assert outputs["moves_left"].shape == (2,)
    assert torch.allclose(outputs["wdl"].softmax(-1).sum(-1), torch.ones(2))
    assert torch.isfinite(outputs["moves_left"]).all()


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
