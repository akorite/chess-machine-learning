import copy
import importlib.util

import numpy as np
import pytest
import torch

from chessfoundation.config import Preset
from chessfoundation.data.encode import RECORD_DTYPE
from chessfoundation.model.student import Student
from chessfoundation.quant import export, qat, verify


def tiny_model():
    return Student(Preset(d_model=32, n_layers=2, n_heads=4, stem_channels=8, stem_blocks=1))


def dataset(n=16):
    records = np.zeros(n, dtype=RECORD_DTYPE)
    for i in range(n):
        records["policy"][i, 0] = 700
        records["policy_prob"][i, 0] = 65535
        records["wdl"][i] = 1
    return records


def test_prepare_and_convert_round_trip():
    torch.manual_seed(0)
    model = tiny_model()
    original = copy.deepcopy(model)
    qat.prepare_qat(model)
    assert qat.is_quantised(model)
    qat.convert(model)
    assert not qat.is_quantised(model)
    with torch.no_grad():
        before = original(torch.zeros(1, 20, 8, 8))
        after = model(torch.zeros(1, 20, 8, 8))
    assert torch.allclose(before["policy"], after["policy"], atol=1e-6)


def test_fake_quant_keeps_gradients():
    x = torch.randn(8, 8, requires_grad=True)
    qat.fake_quant(x).sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_fake_quant_rounds_to_int8_grid():
    x = torch.tensor([0.0, 1.0, -1.0, 0.004])
    quantised = qat.fake_quant(x)
    step = 1.0 / 127
    assert quantised[0] == 0.0
    assert quantised[1] == pytest.approx(1.0)
    assert quantised[2] == pytest.approx(-1.0)
    assert quantised[3] == pytest.approx(step)
    assert torch.allclose(quantised / step, (quantised / step).round())


def test_compare_identical_models_agrees_fully():
    torch.manual_seed(0)
    model = tiny_model()
    report = verify.compare(model, copy.deepcopy(model), dataset(8))
    assert report.move_agreement == pytest.approx(1.0)
    assert report.value_mae_delta == pytest.approx(0.0, abs=1e-6)
    assert report.latency_fp_ms > 0 and report.latency_candidate_ms > 0


def test_compare_detects_disagreement():
    torch.manual_seed(0)
    model = tiny_model()
    other = tiny_model()
    report = verify.compare(model, other, dataset(8))
    assert report.move_agreement < 1.0


@pytest.mark.skipif(importlib.util.find_spec("onnx") is None, reason="onnx not installed")
def test_export_writes_loadable_graph(tmp_path):
    model = tiny_model()
    path = export.export_onnx(model, tmp_path / "model.onnx")
    import onnx

    onnx.checker.check_model(str(path))


def test_tensorrt_build_fails_cleanly_without_runtime(tmp_path):
    if importlib.util.find_spec("tensorrt") is not None:
        pytest.skip("TensorRT installed")
    with pytest.raises(RuntimeError, match="TensorRT"):
        export.build_tensorrt(tmp_path / "none.onnx", tmp_path / "engine.trt")
