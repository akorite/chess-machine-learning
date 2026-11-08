"""Typed run configuration. Unknown keys fail before a VM starts."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Preset:
    d_model: int
    n_layers: int
    n_heads: int
    stem_blocks: int = 6
    stem_channels: int = 64
    pv_plies: int = 6


PRESETS: dict[str, Preset] = {
    "p6m": Preset(d_model=312, n_layers=8, n_heads=8),
    "p20m": Preset(d_model=488, n_layers=12, n_heads=8),
}


@dataclass(frozen=True)
class DataConfig:
    root: str = "data"
    lichess_evals: str = "Lichess/chess-position-evaluations"
    gigafish: str = "lukesalamone/gigafish-3.8b-d10"
    chessbench: str = "google-deepmind/chessbench"
    lc0_chunks: str = ""
    min_depth: int = 24
    min_knodes: int = 100_000
    shard_records: int = 100_000
    max_shards: int = 0


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    lr: float = 3e-4
    warmup_steps: int = 500
    max_steps: int = 100_000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    qat: bool = False
    compile: bool = False
    seed: int = 0
    prior_kl_weight: float = 0.05


@dataclass(frozen=True)
class Phase:
    name: str
    cu: float
    accel: str = "L4"


@dataclass(frozen=True)
class BudgetConfig:
    target_cu: float = 110.0
    ceiling_cu: float = 150.0
    reserve_cu: float = 5.0
    phases: tuple[Phase, ...] = (
        Phase("data", 8.0),
        Phase("throughput", 4.0),
        Phase("pretrain", 45.0),
        Phase("labels", 12.0),
        Phase("distill", 10.0),
        Phase("qat", 4.0),
        Phase("match", 8.0),
        Phase("contingency", 8.0),
    )


@dataclass(frozen=True)
class RunConfig:
    preset: str = "p20m"
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    def __post_init__(self) -> None:
        if self.preset not in PRESETS:
            raise ValueError(f"unknown preset {self.preset!r}; known: {sorted(PRESETS)}")


class ConfigError(ValueError):
    pass


def _check_keys(section: str, raw: dict[str, Any], cls: type) -> None:
    if not is_dataclass(cls):
        return
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"unknown field(s) in [{section}]: {', '.join(unknown)}")


def _build(cls: type, raw: dict[str, Any], section: str) -> Any:
    _check_keys(section, raw, cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        if f.name == "phases":
            kwargs[f.name] = tuple(_build(Phase, p, f"{section}.phases[]") for p in value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> RunConfig:
    raw = tomllib.loads(Path(path).read_text())
    _check_keys("run", raw, RunConfig)
    cfg = RunConfig(
        preset=raw.get("preset", RunConfig.preset),
        data=_build(DataConfig, raw.get("data", {}), "data"),
        train=_build(TrainConfig, raw.get("train", {}), "train"),
        budget=_build(BudgetConfig, raw.get("budget", {}), "budget"),
    )
    return cfg
