"""Build fixed-record cache shards from a public source or a source mix."""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import os
import shutil
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import chess
import numpy as np

from . import chessbench, gigafish, lc0_games, puzzles
from .cache import FORMAT_STAMP, clear_shards, sha256, shard_paths, verify, write_shard
from .cache import MANIFEST as MANIFEST_NAME
from .encode import (
    RECORD_DTYPE,
    move_index,
    record_from_board,
    record_from_eval,
    set_best_move,
)
from .lichess_evals import depth_bands, iter_evals
from .sources import resolve_shards

SOURCE_NAMES = ("lichess", "gigafish", "chessbench", "puzzles", "games")
SOURCE_DEFAULTS = {
    "lichess": "Lichess/chess-position-evaluations",
    "gigafish": "lukesalamone/gigafish-3.8b-d10",
    "chessbench": "google-deepmind/chessbench",
}
DEFAULT_OPTIONS = {
    "min_depth": 24,
    "min_knodes": 100_000,
    "min_rating": 1200,
    "max_rating": 2200,
    "max_plies": 200,
}
PART_BLOCK = 65_536


def _principal(line: str) -> str:
    """First PV move only; deeper plies stay zero and cost no legality check."""
    tokens = line.split()
    return tokens[0] if tokens else ""


def _counted(
    name: str, records: Iterable[np.ndarray], counts: dict[str, int] | None
) -> Iterator[np.ndarray]:
    if counts is None:
        yield from records
        return
    key = f"source:{name}"
    for record in records:
        counts[key] = counts.get(key, 0) + 1
        yield record


def lichess_records(
    source: str,
    min_depth: int,
    min_knodes: int,
    limit_shards: int = 0,
    counts: dict[str, int] | None = None,
) -> Iterator[np.ndarray]:
    return _counted(
        "lichess", _lichess(source, min_depth, min_knodes, limit_shards, counts), counts
    )


def _lichess(
    source: str,
    min_depth: int,
    min_knodes: int,
    limit_shards: int,
    counts: dict[str, int] | None,
) -> Iterator[np.ndarray]:
    for row in iter_evals(
        source, min_depth, min_knodes, deep=True, limit_shards=limit_shards
    ):
        depth = int(
            getattr(row, "depth", 0)
            if hasattr(row, "depth")
            else (row.get("depth") or 0)
        )
        if depth < 20:
            continue
        try:
            record = record_from_eval(row.fen, row.cp, row.mate, _principal(row.line))
        except (ValueError, chess.InvalidMoveError):
            continue
        if counts is not None:
            band = "20to23" if depth < 24 else "24plus"
            counts[band] = counts.get(band, 0) + 1
            knodes = int(
                getattr(row, "knodes", 0)
                if hasattr(row, "knodes")
                else (row.get("knodes") or 0)
            )
            if knodes >= 100_000:
                counts["knodes100k"] = counts.get("knodes100k", 0) + 1
        yield record


def gigafish_records(
    source: str, limit_shards: int = 0, counts: dict[str, int] | None = None
) -> Iterator[np.ndarray]:
    return _counted("gigafish", _gigafish(source, limit_shards), counts)


def _gigafish(source: str, limit_shards: int) -> Iterator[np.ndarray]:
    for row in gigafish.iter_evals(source, limit_shards):
        try:
            yield record_from_eval(row["fen"], row["cp"], row["mate"], "")
        except (ValueError, chess.InvalidMoveError):
            continue


def chessbench_records(
    source: str,
    columns: dict[str, str] | None = None,
    limit_shards: int = 0,
    counts: dict[str, int] | None = None,
) -> Iterator[np.ndarray]:
    return _counted("chessbench", _chessbench(source, columns, limit_shards), counts)


def _chessbench(
    source: str, columns: dict[str, str] | None, limit_shards: int
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
        value = float(row[names["value"]])
        record["wdl"] = chessbench.value_to_wdl(value)
        # ±1 action value ≈ mate; 400 cp per unit keeps value_to_wdl's 0.15 edge near DRAW_CP.
        cp = round(value * 400)
        record["cp"] = cp if board.turn == chess.WHITE else -cp
        yield record


def puzzle_records(
    path: str,
    min_rating: int = 1200,
    max_rating: int = 2200,
    counts: dict[str, int] | None = None,
) -> Iterator[np.ndarray]:
    return _counted("puzzles", _puzzles(path, min_rating, max_rating), counts)


def _puzzles(path: str, min_rating: int, max_rating: int) -> Iterator[np.ndarray]:
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


def game_records(
    paths: list[str], max_plies: int = 200, counts: dict[str, int] | None = None
) -> Iterator[np.ndarray]:
    return _counted("games", _games(paths, max_plies), counts)


def _games(paths: list[str], max_plies: int) -> Iterator[np.ndarray]:
    for board in lc0_games.iter_boards(paths, max_plies):
        yield record_from_board(board)


def source_records(
    name: str,
    path: str,
    options: dict[str, int] | None = None,
    limit_shards: int = 0,
    counts: dict[str, int] | None = None,
) -> Iterator[np.ndarray]:
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    if name == "lichess":
        return lichess_records(
            path, opts["min_depth"], opts["min_knodes"], limit_shards, counts
        )
    if name == "gigafish":
        return gigafish_records(path, limit_shards, counts)
    if name == "chessbench":
        return chessbench_records(path, limit_shards=limit_shards, counts=counts)
    if name == "puzzles":
        return puzzle_records(path, opts["min_rating"], opts["max_rating"], counts)
    if name == "games":
        return game_records([path], opts["max_plies"], counts)
    raise ValueError(f"unknown source {name!r}; choose from {', '.join(SOURCE_NAMES)}")


def mix_specs(values: Iterable[str]) -> list[tuple[str, int]]:
    """Parse `source:count` pairs; count 0 or absent keeps the whole source."""
    specs: list[tuple[str, int]] = []
    for value in values:
        name, _, count = value.partition(":")
        name = name.strip()
        if name not in SOURCE_NAMES:
            raise ValueError(
                f"unknown source {name!r}; choose from {', '.join(SOURCE_NAMES)}"
            )
        specs.append((name, int(count) if count.strip() else 0))
    return specs


def source_paths(values: Iterable[str]) -> tuple[str, dict[str, str]]:
    """Split `[name=]path` values into a bare path and per-source overrides."""
    bare, named = "", {}
    for value in values:
        name, sep, path = value.partition("=")
        if sep:
            named[name] = path
        else:
            bare = value
    return bare, named


def path_for(name: str, bare: str = "", named: dict[str, str] | None = None) -> str:
    path = (named or {}).get(name) or bare or SOURCE_DEFAULTS.get(name, "")
    if not path:
        raise ValueError(f"source {name!r} needs --source-path {name}=<path>")
    return path


def _shards_covering(shards: list[Path], cap: int) -> list[Path]:
    """Stop adding shards once their row counts reach the cap; never more than one over."""
    import pyarrow.parquet as pq

    selected: list[Path] = []
    rows = 0
    for shard in shards:
        if rows >= cap:
            break
        selected.append(shard)
        try:
            rows += int(pq.ParquetFile(shard).metadata.num_rows)
        except (OSError, ValueError):  # unreadable metadata: one shard is enough
            rows = cap
    return selected


def mix_jobs(
    specs: list[tuple[str, int]],
    options: dict[str, int] | None = None,
    bare: str = "",
    named: dict[str, str] | None = None,
    *,
    limit_shards: int = 0,
    scratch: str | Path = ".",
) -> list[dict]:
    """One job per parquet shard, in source then file order, so write order is fixed."""
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    jobs: list[dict] = []
    for name, cap in specs:
        shards = resolve_shards(path_for(name, bare, named), limit_shards)
        if cap:
            shards = _shards_covering(shards, cap)
        for index, shard in enumerate(shards):
            jobs.append(
                {
                    "source": name,
                    "cap": cap,
                    "options": opts,
                    "shard": str(shard),
                    "part": str(
                        Path(scratch) / f"part-{len(jobs):05d}-{index:05d}-{name}.bin"
                    ),
                }
            )
    return jobs


def convert(job: dict) -> tuple[str, str, int, dict[str, int]]:
    """Worker: convert one source shard into a part file owned by this process only."""
    local: dict[str, int] = {}
    records = source_records(job["source"], job["shard"], job["options"], 0, local)
    if job["cap"]:
        records = itertools.islice(records, job["cap"])
    part = Path(job["part"])
    part.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with open(part, "wb") as handle:
        block: list[np.ndarray] = []
        for record in records:
            block.append(record)
            if len(block) >= PART_BLOCK:
                np.stack(block).tofile(handle)
                done += len(block)
                block = []
        if block:
            np.stack(block).tofile(handle)
            done += len(block)
    local.pop(f"source:{job['source']}", None)
    return job["source"], str(part), done, local


class _Writer:
    """Append records to `shard-NNNNN.bin` files, rewriting the manifest after each shard."""

    def __init__(
        self,
        out: str | Path,
        shard_size: int = 100_000,
        max_records: int = 0,
        counts: dict[str, int] | None = None,
        append: bool = False,
    ) -> None:
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.shard_size = max(1, int(shard_size))
        self.max_records = int(max_records)
        self.counts = counts if counts is not None else {}
        self.manifest = self.out / MANIFEST_NAME
        self.entries: dict[str, dict] = {}
        self.prior: dict[str, int] = {}
        self.prior_total = 0
        self.shard_index = 0
        self.added = 0
        self.pieces: list[np.ndarray] = []
        self.fill = 0
        self._resume(append)

    @property
    def total(self) -> int:
        return self.prior_total + self.added

    @property
    def full(self) -> bool:
        return bool(self.max_records) and self.total >= self.max_records

    def _resume(self, append: bool) -> None:
        if append and self.manifest.exists():
            try:
                prior = json.loads(self.manifest.read_text())
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                print(
                    f"warning: could not read prior manifest {self.manifest}: {exc}",
                    file=sys.stderr,
                )
            else:
                self.entries.update(prior.get("shards", {}))
                prior_counts = prior.get("counts", {})
                self.prior_total = int(prior_counts.get("records", 0))
                self.prior = {
                    k: int(v) for k, v in prior_counts.items() if k != "records"
                }
                indices = [
                    int(path.stem.rsplit("-", 1)[-1])
                    for path in shard_paths(self.out)
                    if path.stem.rsplit("-", 1)[-1].isdigit()
                ]
                if indices:
                    self.shard_index = max(indices) + 1
        elif not append:
            clear_shards(self.out)

    def add(self, block: np.ndarray) -> int:
        """Append a block of records, flushing every complete shard; returns records kept."""
        if self.full:
            return 0
        records = np.ascontiguousarray(block, dtype=RECORD_DTYPE)
        if self.max_records:
            records = records[: self.max_records - self.total]
        offset = 0
        while offset < len(records):
            take = min(len(records) - offset, self.shard_size - self.fill)
            self.pieces.append(records[offset : offset + take])
            self.fill += take
            self.added += take
            offset += take
            if self.fill == self.shard_size:
                self._flush()
        return offset

    def _flush(self) -> None:
        if not self.fill:
            return
        block = self.pieces[0] if len(self.pieces) == 1 else np.concatenate(self.pieces)
        self.pieces = []
        self.fill = 0
        path = write_shard(self.out / f"shard-{self.shard_index:05d}.bin", block)
        self.entries[path.name] = {"sha256": sha256(path), "records": len(block)}
        self.shard_index += 1
        self.write_manifest()

    def write_manifest(self) -> None:
        merged = dict(self.prior)
        for key, value in self.counts.items():
            merged[key] = merged.get(key, 0) + value
        merged["records"] = sum(
            int(entry["records"]) for entry in self.entries.values()
        )
        self.manifest.write_text(
            json.dumps(
                {"format": FORMAT_STAMP, "counts": merged, "shards": self.entries},
                indent=2,
                sort_keys=True,
            )
        )

    def close(self) -> dict[str, int]:
        self._flush()
        if self.total == 0:
            raise ValueError("no records produced")
        # counts are merged by the caller after a shard flush, so restate them once at the end
        self.write_manifest()
        return {"records": self.total, "shards": self.shard_index}


def build(
    records: Iterable[np.ndarray],
    out: str | Path,
    shard_size: int = 100_000,
    max_records: int = 0,
    counts: dict[str, int] | None = None,
    append: bool = False,
) -> dict[str, int]:
    writer = _Writer(out, shard_size, max_records, counts, append)
    stream = records
    if max_records:
        stream = itertools.islice(records, max(0, max_records - writer.total))
    for record in stream:
        writer.add(np.asarray(record, dtype=RECORD_DTYPE).reshape(1))
    return writer.close()


def build_mix(
    specs: list[tuple[str, int]],
    out: str | Path,
    bare: str = "",
    named: dict[str, str] | None = None,
    options: dict[str, int] | None = None,
    workers: int = 0,
    shard_size: int = 100_000,
    max_records: int = 0,
    append: bool = False,
    limit_shards: int = 0,
) -> dict[str, int]:
    """Convert source shards in worker processes; the parent writes shards in order."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / "parts"
    shutil.rmtree(scratch, ignore_errors=True)  # parts from an interrupted run
    scratch.mkdir(parents=True, exist_ok=True)
    jobs = mix_jobs(
        specs, options, bare, named, limit_shards=limit_shards, scratch=scratch
    )
    if not jobs:
        raise ValueError("no source shards found")
    caps = dict(specs)
    counts: dict[str, int] = {}
    writer = _Writer(out, shard_size, max_records, counts, append)
    written = dict.fromkeys(caps, 0)
    workers = workers or os.cpu_count() or 1
    # spawn: forking a parent that holds pyarrow/numpy threads can deadlock the children.
    pool = (
        multiprocessing.get_context("spawn").Pool(min(workers, len(jobs)))
        if workers > 1 and len(jobs) > 1
        else None
    )
    try:
        # shortcut: each worker stops at its source's cap, but not at what earlier shards of
        # that source already took, so per-source totals are exact while depth bands may count
        # up to one shard's tail past the cap; give workers the remaining count if that matters.
        results = pool.imap(convert, jobs) if pool is not None else map(convert, jobs)
        for name, part, done, bands in results:
            cap = caps[name]
            keep = done if not cap else min(done, cap - written[name])
            if keep > 0:
                block = np.fromfile(part, dtype=RECORD_DTYPE, count=keep)
                added = writer.add(block)
                written[name] += added
                key = f"source:{name}"
                counts[key] = counts.get(key, 0) + added
                for band, value in bands.items():
                    counts[band] = counts.get(band, 0) + value
                writer.write_manifest()
            Path(part).unlink()
            if writer.full:
                break
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    result = writer.close()
    shutil.rmtree(scratch, ignore_errors=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=list(SOURCE_NAMES))
    parser.add_argument(
        "--source-path", action="append", default=[], metavar="[NAME=]PATH"
    )
    parser.add_argument("--mix", action="append", default=[], metavar="SOURCE:COUNT")
    parser.add_argument("--workers", type=int, default=0)
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

    bare, named = source_paths(args.source_path)
    specs = mix_specs(args.mix)

    if args.verify:
        errors = verify(args.out)
        print("\n".join(errors) if errors else "ok")
        return 1 if errors else 0
    if args.count_only:
        name = specs[0][0] if specs else args.source
        if not name:
            parser.error("--count-only needs --source or --mix")
        print(depth_bands(path_for(name, bare, named), args.limit_shards))
        return 0

    options = {
        "min_depth": args.min_depth,
        "min_knodes": args.min_knodes,
        "min_rating": args.min_rating,
        "max_rating": args.max_rating,
        "max_plies": 200,
    }
    if specs:
        result = build_mix(
            specs,
            args.out,
            bare=bare,
            named=named,
            options=options,
            workers=args.workers,
            shard_size=args.shard_size,
            max_records=args.max_records,
            append=args.append,
            limit_shards=args.limit_shards,
        )
    else:
        if not args.source:
            parser.error("--source or --mix is required")
        counts: dict[str, int] = {}
        records = source_records(
            args.source,
            path_for(args.source, bare, named),
            options,
            args.limit_shards,
            counts,
        )
        result = build(
            records,
            args.out,
            args.shard_size,
            args.max_records,
            counts,
            append=args.append,
        )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
