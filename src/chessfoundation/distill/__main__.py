"""Distillation round driver: play -> label -> reanalyse -> retrain -> gate.

One invocation = one on-policy round. `current.pt` in --out is the accepted
checkpoint; `rounds.json` records every round's score and gate verdict.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import chess.engine

from ..budget import BudgetExceeded, check_run, read_colab_state
from ..config import RunConfig, TrainConfig
from ..engine.uci import EngineSpec
from ..model.presets import build_student
from ..teacher.lc0_oracle import Lc0Oracle, OracleError, StockfishOracle
from ..train.checkpoint import load, save
from .rounds import apply_gate, gate_match, play_and_collect, run_round


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--white", nargs="+", help="white engine command (default: student)")
    parser.add_argument("--black", nargs="+", help="black engine command (default: student)")
    parser.add_argument("--teacher", choices=["stockfish", "lc0"], default="stockfish")
    parser.add_argument("--teacher-engine", nargs="+", required=True)
    parser.add_argument("--teacher-depth", type=int, default=22)
    parser.add_argument("--teacher-nodes", type=int, default=64)
    parser.add_argument("--teacher-multipv", type=int, default=4)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--movetime-ms", type=int, default=400)
    parser.add_argument("--sims", type=int, default=512)
    parser.add_argument("--preset", default="p6m", choices=["p6m", "p20m"])
    parser.add_argument("--init", default="", help="checkpoint the round starts from")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--prior-kl-weight", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True, help="run directory for shards/checkpoints/rounds.json")
    parser.add_argument("--replay-dir", action="append", default=[], help="previous round data dir (repeatable)")
    parser.add_argument("--reanalyse", type=int, default=1000, help="replay sample re-labeled each round (0 = off)")
    parser.add_argument("--gate-engine", nargs="+", default=None, help="rung opponent for the round gate; omitted = always accept")
    parser.add_argument("--gate-games", type=int, default=4)
    parser.add_argument("--gate-movetime-ms", type=int, default=400)
    parser.add_argument("--gate-nodes", type=int, default=None)
    parser.add_argument("--book", default="", help="polyglot opening book for play and gate")
    parser.add_argument("--openings-file", default="", help="UCI opening lines, one per row")
    parser.add_argument("--accelerator", default="", help="project CU via check_run before GPU work")
    parser.add_argument("--hours", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.accelerator:
        try:
            state = read_colab_state()
        except BudgetExceeded as exc:  # colab CLI absent off-Colab; a real refusal comes from check_run
            print(f"warning: budget state unavailable, skipping CU gate: {exc}", file=sys.stderr)
        else:
            check_run(RunConfig().budget, state, args.accelerator, args.hours, phase="distill")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / "rounds.json"
    history: list[dict] = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    round_index = len(history)
    round_dir = out_dir / f"round-{round_index:04d}"

    openings: tuple[str, ...] = ()
    if args.openings_file:
        openings = tuple(line.strip() for line in Path(args.openings_file).read_text().splitlines() if line.strip())

    model = build_student(args.preset)
    if args.init:
        load(args.init, model)

    def student_cmd(checkpoint: str) -> list[str]:
        cmd = [sys.executable, "-m", "chessfoundation.engine.model_engine",
               "--checkpoint", checkpoint, "--preset", args.preset,
               "--simulations", str(args.sims), "--device", args.device]
        if args.book:
            cmd += ["--book", args.book]
        return cmd

    start_ckpt = args.init or str(out_dir / "init.pt")
    if not args.init:
        save(start_ckpt, model, meta={"stage": "distill-init"})

    white_spec = EngineSpec(args.white or student_cmd(start_ckpt))
    black_spec = EngineSpec(args.black or student_cmd(start_ckpt))
    teacher_path: str | list[str] = args.teacher_engine[0] if len(args.teacher_engine) == 1 else args.teacher_engine
    oracle = (
        StockfishOracle(teacher_path, depth=args.teacher_depth, multipv=args.teacher_multipv)
        if args.teacher == "stockfish"
        else Lc0Oracle(teacher_path, nodes=args.teacher_nodes, multipv=args.teacher_multipv)
    )

    white, black = white_spec.open(), black_spec.open()
    try:
        positions = play_and_collect(
            white, black, games=args.games, movetime_ms=args.movetime_ms,
            openings=openings, rng=random.Random(args.seed),
        )
    finally:
        for engine in (white, black):
            try:
                engine.quit()
            except chess.engine.EngineError:  # a dead play engine must not mask round results
                pass
    if not positions:
        print("round aborted: collection produced no positions", file=sys.stderr)
        return 1

    cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=min(500, args.steps),
        max_steps=args.steps,
        prior_kl_weight=args.prior_kl_weight,
        seed=args.seed,
    )
    try:
        with oracle:
            analyse = lambda board: (
                (result := oracle.analyse(board)).value, result.policy
            )
            metrics = run_round(
                model, positions, analyse, cfg, round_dir,
                replay_dirs=tuple(args.replay_dir),
                device=args.device,
                reanalyse_sample=args.reanalyse,
            )
    except (OracleError, OSError, RuntimeError, ValueError) as exc:
        print(f"round aborted: {exc}", file=sys.stderr)
        return 1

    round_ckpt = save(round_dir / "checkpoint.pt", model,
                      meta={"stage": "distill", "round": round_index, "metrics": metrics})
    score = None
    gate_error = None
    if args.gate_engine:
        try:
            score = gate_match(
                EngineSpec(student_cmd(str(round_ckpt))), EngineSpec(args.gate_engine),
                games=args.gate_games, movetime_ms=args.gate_movetime_ms,
                nodes=args.gate_nodes, openings=openings,
                pgn_path=str(round_dir / "gate.pgn"), seed=args.seed,
            )
        except Exception as exc:  # noqa: BLE001 - a failed gate rejects, never promotes
            gate_error = f"{type(exc).__name__}: {exc}"
    record = apply_gate(out_dir, round_index, round_ckpt, score, history, gate_error=gate_error)
    record["metrics"] = metrics
    history.append(record)
    ledger_path.write_text(json.dumps(history, indent=2))
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
