import pytest

from chessfoundation.config import PRESETS, ConfigError, load_config


def write(tmp_path, text):
    path = tmp_path / "run.toml"
    path.write_text(text)
    return path


def test_presets_carry_required_shape():
    assert PRESETS["p6m"].pv_plies == 6
    assert PRESETS["p20m"].d_model > PRESETS["p6m"].d_model


def test_loads_nested_config(tmp_path):
    cfg = load_config(
        write(
            tmp_path,
            """
preset = "p6m"
[train]
batch_size = 64
lr = 1e-4
[budget]
target_cu = 110.0
ceiling_cu = 150.0
""",
        )
    )
    assert cfg.preset == "p6m"
    assert cfg.train.batch_size == 64
    assert cfg.train.lr == pytest.approx(1e-4)
    assert cfg.budget.ceiling_cu == 150.0


def test_unknown_top_level_field_fails(tmp_path):
    with pytest.raises(ConfigError, match="unknown field"):
        load_config(write(tmp_path, 'preset = "p20m"\nbudgett = 3\n'))


def test_unknown_nested_field_fails(tmp_path):
    with pytest.raises(ConfigError, match="unknown field"):
        load_config(write(tmp_path, "[train]\nbatc_size = 64\n"))


def test_unknown_preset_fails():
    from chessfoundation.config import RunConfig

    with pytest.raises(ValueError, match="unknown preset"):
        RunConfig(preset="p999m")
