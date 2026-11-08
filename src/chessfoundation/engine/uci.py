"""UCI engine handling: rung options, validation, and a local game runner."""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import chess.engine


@dataclass(frozen=True)
class EngineSpec:
    path: str | list[str]
    options: dict[str, object] = field(default_factory=dict)

    def open(self, timeout: float = 30.0) -> chess.engine.SimpleEngine:
        engine = chess.engine.SimpleEngine.popen_uci(self.path, timeout=timeout)
        if self.options:
            try:
                validate_rung(engine, dict(self.options))
            except Exception:
                engine.quit()
                raise
        return engine


def rung_options(uci_elo: int, threads: int = 1, hash_mb: int = 16) -> dict[str, object]:
    return {
        "UCI_LimitStrength": True,
        "UCI_Elo": int(uci_elo),
        "Threads": int(threads),
        "Hash": int(hash_mb),
    }


def validate_rung(engine: chess.engine.SimpleEngine, options: dict[str, object]) -> None:
    """Fail when a rung cannot be pinned, instead of silently playing full strength."""
    for name in options:
        if name not in engine.options:
            raise ValueError(f"engine does not expose required option {name!r}")
    engine.configure(dict(options))

