"""ONNX export of the inference graph (policy, promotion, WDL only)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ..data.encode import PLANES


class InferenceGraph(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, planes: torch.Tensor):
        outputs = self.model(planes)
        return outputs["policy"], outputs["promo"], outputs["wdl"]


def export_onnx(model: nn.Module, path: str | Path, opset: int = 17) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = InferenceGraph(model).eval()
    dummy = torch.zeros(1, PLANES, 8, 8)
    torch.onnx.export(
        graph,
        dummy,
        path,
        input_names=["planes"],
        output_names=["policy", "promo", "wdl"],
        dynamic_axes={"planes": {0: "batch"}, "policy": {0: "batch"}, "promo": {0: "batch"}, "wdl": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    return path


def build_tensorrt(onnx_path: str | Path, engine_path: str | Path) -> Path:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("TensorRT is required to build an engine") from exc
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError(f"TensorRT failed to parse {onnx_path}")
    engine = builder.build_serialized_network(network, builder.create_builder_config())
    Path(engine_path).write_bytes(engine)
    return Path(engine_path)
