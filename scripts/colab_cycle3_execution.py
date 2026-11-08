"""Cycle-3 end-to-end run: corpus build, training, evaluation.

Kept in one process so the VM is never idle between stages: Colab reclaims a
runtime whose kernel goes quiet, and the corpus build plus the training run are
long enough that a gap loses the boot disk.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

CODE = "/content/code"
SRC = f"{CODE}/src"
CACHE = "/content/cache/mix"
CHECKPOINTS = "/content/checkpoints"
RESULTS = "/content/results"
STOCKFISH = "/content/bin/stockfish/stockfish-ubuntu-x86-64-avx2"  # resolved by ensure_engine()
BOOK = f"{CODE}/data/book.bin"
SOURCE_PATHS = [
    "--source-path", "lichess=/content/data/lichess_evals",
    "--source-path", "gigafish=/content/data/gigafish",
]
ENV = {**os.environ, "PYTHONPATH": SRC}
if os.path.isdir(f"{SRC}/chessfoundation") and SRC not in sys.path:
    sys.path.insert(0, SRC)


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with open("/content/cycle3.log", "a") as handle:
        handle.write(line + "\n")


def run(args: list[str], timeout: int = 86400) -> int:
    log("$ " + " ".join(args))
    started = time.time()
    proc = subprocess.run(args, capture_output=True, text=True, env=ENV, timeout=timeout, check=False)
    log((proc.stdout or "")[-2000:])
    if proc.returncode:
        log("STDERR " + (proc.stderr or "")[-2000:])
    log(f"rc={proc.returncode} wall={round(time.time() - started)}s")
    return proc.returncode


def ensure_engine() -> str:
    """Locate a Stockfish binary; the release download is flaky, apt is not."""
    import shutil

    if os.path.exists(STOCKFISH):
        return STOCKFISH
    found = shutil.which("stockfish")
    if found:
        return found
    subprocess.run(["apt-get", "update", "-qq"], check=False)
    subprocess.run(["apt-get", "install", "-y", "-qq", "stockfish"], check=False)
    return shutil.which("stockfish") or STOCKFISH


def stage_corpus() -> int:
    if os.path.exists(f"{CACHE}/manifest.json"):
        log("corpus already present")
        return 0
    return run(
        [
            sys.executable, "-m", "chessfoundation.data.build_cache",
            "--out", CACHE, "--shard-size", "1000000",
            "--mix", "lichess:60000000", "--mix", "gigafish:100000000",
            *SOURCE_PATHS, "--workers", "10",
        ]
    )


def stage_train() -> int:
    from chessfoundation.model.presets import build_student
    from chessfoundation.train import throughput

    model = build_student("p6m").to("cuda")
    measured = throughput.measure(model, batch_size=1024, device="cuda", warmup=5, steps=20, bf16=True)
    log(f"throughput p6m bs=1024 {measured.positions_per_second:,.0f} pos/s "
        f"batch1 {measured.latency_ms_batch1:.1f} ms")
    budget = float(os.environ.get("TRAIN_SECONDS", "3600"))
    batch = 1024
    steps = max(1000, int(measured.positions_per_second * budget / batch))
    log(f"training {steps:,} steps (~{steps * batch / 1e6:.0f}M positions)")
    code = run(
        [
            sys.executable, "-m", "chessfoundation.train.pretrain",
            "--stage", "deep", "--data", CACHE, "--preset", "p6m",
            "--steps", str(steps), "--batch-size", str(batch), "--lr", "3e-4",
            "--device", "cuda", "--save-every", "0",
            "--out", f"{CHECKPOINTS}/deep3.pt",
        ]
    )
    if code == 0 or os.path.exists(f"{CHECKPOINTS}/deep3.pt"):
        with open(f"{CHECKPOINTS}/deep3.metrics.json") as handle:
            log("final metrics " + handle.read()[:600])
    return code


def stage_eval() -> int:
    student = "/content/bin/student3.sh"
    with open(student, "w") as handle:
        handle.write(
            "#!/bin/bash\n"
            f"export PYTHONPATH={SRC}\n"
            f"exec {sys.executable} -m chessfoundation.engine.model_engine "
            f"--checkpoint {CHECKPOINTS}/deep3.pt --preset p6m --simulations 128 "
            "--search gumbel --device cuda \"$@\"\n"
        )
    os.chmod(student, 0o755)
    opponent = ensure_engine()
    if not os.path.exists(opponent):
        log("no Stockfish binary; skipping matches")
        return 1
    summary = {}
    for rung in (1320, 1500, 2000):
        out = f"{RESULTS}/ladder-{rung}.md"
        run(
            [
                sys.executable, "-m", "chessfoundation.eval.ladder",
                "--engine", student, "--opponent", opponent,
                "--rungs", str(rung), "--book", BOOK, "--games", "40",
                "--movetime", "100", "--pgn", f"{RESULTS}/ladder-{rung}", "--out", out,
            ]
        )
        if os.path.exists(out):
            with open(out) as handle:
                summary[f"ladder_{rung}"] = handle.read()
    with open(f"{RESULTS}/cycle3-summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    return 0


def main() -> int:
    os.makedirs(CHECKPOINTS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs("/content/bin", exist_ok=True)
    os.makedirs(CODE, exist_ok=True)
    if not os.path.isdir(f"{SRC}/chessfoundation"):
        subprocess.run(["tar", "xzf", "/c3code.tgz", "-C", CODE], check=True)
    sys.path.insert(0, SRC)
    log("=== corpus ===")
    if stage_corpus():
        return 1
    run([sys.executable, "-m", "chessfoundation.data.build_cache", "--verify", "--out", CACHE])
    with open(f"{CACHE}/manifest.json") as handle:
        log("manifest " + json.dumps(json.loads(handle.read())["counts"])[:400])
    log("=== train ===")
    if stage_train():
        return 1
    log("=== eval ===")
    stage_eval()
    log("CYCLE3 DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
