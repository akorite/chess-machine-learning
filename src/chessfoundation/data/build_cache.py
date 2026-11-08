"""Build fixed-record cache shards from a public source."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import chess
import numpy as np

from . import chessbench, gigafish, lc0_games, puzzles
from .cache import MANIFEST as MANIFEST_NAME
from .cache import clear_shards, sha256, verify, write_shard
from .encode import move_index, record_from_board, record_from_eval, set_best_move
from .lichess_evals import depth_bands, iter_evals


def _safe(records: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
    yield from records


def lichess_records(
    source: str,
    min_depth: int,
    min_knodes: int,
    limit_shards: int = 0,
    counts: dict[str, int] | None = None,
) -> Iterator[np.ndarray]:
    for row in iter_evals(source, min_depth, min_knodes, deep=True, limit_shards=limit_shards):
        if counts is not None:
            depth = int(getattr(row, "depth", 0) if hasattr(row, "depth") else (row.get("depth") or 0))
            if depth < 20:
                band = "lt20"
            elif depth < 24:
                band = "20to23"
            else:
                band = "24plus"
            counts[band] = counts.get(band, 0) + 1
            knodes = int(getattr(row, "knodes", 0) if hasattr(row, "knodes") else (row.get("knodes") or 0))
            if knodes >= 100_000:
                counts["knodes100k"] = counts.get("knodes100k", 0) + 1
        try:
            yield record_from_eval(row.fen, row.cp, row.mate, row.line)
        except (ValueError, chess.InvalidMoveError):
            continue


def gigafish_records(source: str, limit_shards: int = 0) -> Iterator[np.ndarray]:
    for row in gigafish.iter_evals(source, limit_shards):
        try:
            yield record_from_eval(row["fen"], row["cp"], row["mate"], "")
        except (ValueError, chess.InvalidMoveError):
            continue


def chessbench_records(
    source: str, columns: dict[str, str] | None = None, limit_shards: int = 0
) -> Iterator[np.ndarray]:
    names = columns or chessbench.DEFAULT_COLUMNS
    for row in chessbench.iter_actions(source, names, limit_shards):
        try:
            board = chess.Board(row[names["fen"]])
            move = chess.Move.from_uci(row[names["move"]])
        except (ValueError, chess.InvalidMoveError):
            continue
        if not board.is_legal(move):
            continue
        record = record_from_board(board)
        set_best_move(record, move)
        record["wdl"] = chessbench.value_to_wdl(float(row[names["value"]]))
        yield record


def puzzle_records(path: str, min_rating: int = 1200, max_rating: int = 2200) -> Iterator[np.ndarray]:
    for puzzle in puzzles.iter_puzzles(path, min_rating, max_rating):
        if len(puzzle.moves) < 2:
            continue
        try:
            board = chess.Board(puzzle.fen)
            board.push_uci(puzzle.moves[0])
            solution = board.parse_uci(puzzle.moves[1])
        except (ValueError, chess.InvalidMoveError):
            continue
        record = record_from_board(board)
        record["policy"][0] = move_index(solution)
        record["policy_prob"][0] = 65535
        yield record


def game_records(paths: list[str], max_plies: int = 200) -> Iterator[np.ndarray]:
    for board in lc0_games.iter_boards(paths, max_plies):
        yield record_from_board(board)


def build(
    records: Iterable[np.ndarray],
    out: str | Path,
    shard_size: int = 100_000,
    max_records: int = 0,
    counts: dict[str, int] | None = None,
    append: bool = False,
) -> dict[str, int]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / MANIFEST_NAME
    entries: dict[str, dict] = {}
    counts_accum: dict[str, int] = {}
    shard_index = 0
    prior_records = 0

    if append and manifest_path.exists():
        try:
            prior_data = json.loads(manifest_path.read_text())
            entries.update(prior_data.get("shards", {}))
            prior_counts = prior_data.get("counts", {})
            prior_records = int(prior_counts.get("records", 0))
            for k, v in prior_counts.items():
                if k != "records":
                    counts_accum[k] = counts_accum.get(k, 0) + v
            indices = [
                int(p.stem.split("-")[-1])
                for p in out.glob("shard-*.bin")
                if p.stem.split("-")[-1].isdigit()
            ]
            if indices:
                shard_index = max(indices) + 1
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            import sys
            print(f"warning: could not read prior manifest {manifest_path}: {exc}", file=sys.stderr)
    elif not append:
        clear_shards(out)

    buffer: list[np.ndarray] = []
    added = 0

    def flush() -> None:
        nonlocal shard_index, buffer
        if not buffer:
            return
        path = write_shard(out / f"shard-{shard_index:05d}.bin", np.stack(buffer))
        entries[path.name] = {
            "sha256": sha256(path),
            "records": len(buffer),
        }
        buffer = []
        shard_index += 1
        out_counts = dict(counts_accum)
        if counts is not None:
            for k, v in counts.items():
                out_counts[k] = out_counts.get(k, 0) + v
        out_counts["records"] = prior_records + added
        manifest_path.write_text(
            json.dumps({"counts": out_counts, "shards": entries}, indent=2, sort_keys=True)
        )

    for record in records:
        buffer.append(record)
        added += 1
        if len(buffer) >= shard_size:
            flush()
        if max_records and (prior_records + added) >= max_records:
            break
    flush()
    total = prior_records + added
    if total == 0:
        raise ValueError("no records produced")
    return {"records": total, "shards": shard_index}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["lichess", "gigafish", "chessbench", "puzzles", "games"])
    parser.add_argument("--source-path", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--min-depth", type=int, default=24)
    parser.add_argument("--min-knodes", type=int, default=100_000)
    parser.add_argument("--min-rating", type=int, default=1200)
    parser.add_argument("--max-rating", type=int, default=2200)
    parser.add_argument("--limit-shards", type=int, default=0)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args(argv)

    if args.verify:
        errors = verify(args.out)
        print("\n".join(errors) if errors else "ok")
        return 1 if errors else 0
    if args.count_only:
        print(depth_bands(args.source_path or args.source, args.limit_shards))
        return 0
    if not args.source:
        parser.error("--source is required")

    counts = {}
    if args.source == "lichess":
        counts = {}
        records = lichess_records(
            args.source_path or "Lichess/chess-position-evaluations",
            args.min_depth,
            args.min_knodes,
            args.limit_shards,
            counts,
        )
    elif args.source == "gigafish":
        records = gigafish_records(
            args.source_path or "lukesalamone/gigafish-3.8b-d10", args.limit_shards
        )
    elif args.source == "chessbench":
        records = chessbench_records(
            args.source_path or "google-deepmind/chessbench",
            limit_shards=args.limit_shards,
        )
    elif args.source == "puzzles":
        records = puzzle_records(args.source_path, args.min_rating, args.max_rating)
    else:
        records = game_records([args.source_path])
    result = build(records, args.out, args.shard_size, args.max_records, counts, append=args.append)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
