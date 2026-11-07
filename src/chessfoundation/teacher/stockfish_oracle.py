"""Stockfish oracle (depth-bound) and the shared value type."""

from __future__ import annotations

from .lc0_oracle import OracleResult, StockfishOracle, UciOracle, value_from_score

__all__ = ["OracleResult", "StockfishOracle", "UciOracle", "value_from_score"]
