"""Preset construction and parameter budgets."""

from __future__ import annotations

from ..config import PRESETS
from .student import Student, parameter_count

TARGETS = {"p6m": 6_000_000, "p20m": 20_000_000}
TOLERANCE = 0.05


def build_student(name: str, pure_transformer: bool = False) -> Student:
    preset = PRESETS[name]
    if preset.n_heads < 8:
        raise ValueError(
            f"{name} has {preset.n_heads} heads; the 64-token trunk needs >= 8"
        )
    model = Student(preset, pure_transformer=pure_transformer)
    target = TARGETS[name]
    count = parameter_count(model)
    if not pure_transformer and abs(count - target) > TOLERANCE * target:
        raise ValueError(f"{name} has {count:,} parameters, outside 5% of {target:,}")
    return model
