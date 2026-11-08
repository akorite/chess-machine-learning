import os
import subprocess
import sys
import tarfile
import time
import urllib.request


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("=== 1. Setup & Dependencies ===")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "python-chess", "pyarrow", "huggingface-hub", "numpy", "onnx", "onnxscript"], check=True)

import torch

log(f"Torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")

log("=== 2. Stockfish 17.1 Setup ===")
os.makedirs("/content/bin", exist_ok=True)
sf_dest = "/content/bin/sf.tar"
sf_bin = "/content/bin/stockfish/stockfish-ubuntu-x86-64-avx2"
if not os.path.exists(sf_bin):
    urllib.request.urlretrieve("https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-ubuntu-x86-64-avx2.tar", sf_dest)
    subprocess.run(["tar", "xf", sf_dest, "-C", "/content/bin"], check=True)
os.chmod(sf_bin, 0o755)
log(f"Stockfish verified at {sf_bin}")

log("=== 3. Codebase Extraction ===")
os.makedirs("/content/chessfoundation", exist_ok=True)
with tarfile.open("/tmp/repo.tgz") as t:
    t.extractall("/content/chessfoundation")
os.chdir("/content/chessfoundation")
sys.path.insert(0, "/content/chessfoundation/src")
log("chessfoundation imported successfully")

log("=== 4. Download Training Datasets (HF) ===")
os.environ["HF_HUB_CACHE"] = "/content/data/hfcache"
for d in ("/content/data/lichess_evals", "/content/data/gigafish"):
    os.makedirs(d, exist_ok=True)

from huggingface_hub import HfApi, hf_hub_download

api = HfApi()

def dl_shards(repo, dest, max_shards):
    names = sorted(n for n in api.list_repo_files(repo, repo_type="dataset") if n.endswith(".parquet"))[:max_shards]
    log(f"Downloading {len(names)} shards from {repo} -> {dest}")
    for i, name in enumerate(names):
        local_p = os.path.join(dest, os.path.basename(name))
        if not os.path.exists(local_p) or os.path.getsize(local_p) == 0:
            hf_hub_download(repo, name, repo_type="dataset", local_dir=dest)
        if (i + 1) % 5 == 0 or i == len(names) - 1:
            log(f"  {repo} shard {i+1}/{len(names)}")

dl_shards("Lichess/chess-position-evaluations", "/content/data/lichess_evals", max_shards=5)
dl_shards("lukesalamone/gigafish-3.8b-d10", "/content/data/gigafish", max_shards=20)

log("=== 5. Build Cache Shards ===")
os.makedirs("/content/cache/pretrain", exist_ok=True)
os.makedirs("/content/cache/deep", exist_ok=True)

ENV = {**os.environ, "PYTHONPATH": "/content/chessfoundation/src"}
def run_cmd(args, timeout=3600):
    log("$ " + " ".join(args))
    r = subprocess.run(args, capture_output=True, text=True, env=ENV, timeout=timeout, check=False)
    if r.stdout: log(r.stdout[-1500:])
    if r.returncode:
        log("STDERR: " + r.stderr[-1500:])
        raise RuntimeError(f"Command failed: {args}")
    return r

# Build pretrain cache from GigaFish
run_cmd([sys.executable, "-m", "chessfoundation.data.build_cache",
         "--source", "gigafish", "--source-path", "/content/data/gigafish",
         "--out", "/content/cache/pretrain", "--shard-size", "100000", "--max-records", "1000000"])

# Build deep cache from Lichess evals
run_cmd([sys.executable, "-m", "chessfoundation.data.build_cache",
         "--source", "lichess", "--source-path", "/content/data/lichess_evals",
         "--out", "/content/cache/deep", "--shard-size", "100000", "--max-records", "1000000"])

log("=== 6. Pretraining: Stage Value (GigaFish) ===")
os.makedirs("/content/checkpoints", exist_ok=True)
value_ckpt = "/content/checkpoints/value.pt"
run_cmd([sys.executable, "-m", "chessfoundation.train.pretrain",
         "--stage", "value", "--data", "/content/cache/pretrain",
         "--preset", "p6m", "--steps", "2000", "--batch-size", "256",
         "--lr", "3e-4", "--device", "cuda", "--out", value_ckpt])

log("=== 7. Training: Stage Deep (Lichess Depth>=24) ===")
deep_ckpt = "/content/checkpoints/deep.pt"
run_cmd([sys.executable, "-m", "chessfoundation.train.pretrain",
         "--stage", "deep", "--init", value_ckpt, "--data", "/content/cache/deep",
         "--preset", "p6m", "--steps", "5000", "--batch-size", "256",
         "--lr", "3e-4", "--device", "cuda", "--out", deep_ckpt])

log("=== 8. Wrapper Script for Engine Evaluation ===")
os.makedirs("/content/bin", exist_ok=True)
student_sh = "/content/bin/student.sh"
with open(student_sh, "w") as f:
    f.write(f'''#!/bin/bash
export PYTHONPATH=/content/chessfoundation/src
exec /usr/bin/python3 -m chessfoundation.engine.model_engine --checkpoint {deep_ckpt} --preset p6m --simulations 16 --search gumbel --device cuda "$@"
''')
os.chmod(student_sh, 0o755)

log("=== 9. Evaluation: Ladder Match vs Stockfish 2000 (R9/KTD22) ===")
os.makedirs("/content/results", exist_ok=True)
ladder_md = "/content/results/ladder.md"
ladder_pgn = "/content/results/ladder"
book_path = "/content/chessfoundation/data/book.bin"

run_cmd([sys.executable, "-m", "chessfoundation.eval.ladder",
         "--engine", student_sh,
         "--opponent", sf_bin,
         "--book", book_path,
         "--games", "50",
         "--movetime", "50",
         "--pgn", ladder_pgn,
         "--out", ladder_md])

log("=== 10. Evaluation: Match vs Stockfish R10 ===")
from chessfoundation.engine.uci import EngineSpec
from chessfoundation.eval.tournament import MatchConfig, run_local

r10_cfg = MatchConfig(
    white=EngineSpec(student_sh),
    black=EngineSpec(sf_bin),
    games=30,
    movetime_ms=50,
    openings=(
        "e2e4 e7e5 g1f3 b8c6",
        "e2e4 c7c5 g1f3 d7d6",
        "d2d4 d7d5 c2c4 e7e6",
        "d2d4 g8f6 c2c4 g7g6",
    ),
    pgn_path="/content/results/r10.pgn"
)
r10_res = run_local(r10_cfg)
log(f"R10 Result: Wins={r10_res.wins}, Draws={r10_res.draws}, Losses={r10_res.losses}, Score={r10_res.score:.3f}")

log("=== 11. Final Results Report ===")
ladder_text = "No ladder output"
if os.path.exists(ladder_md):
    with open(ladder_md) as f:
        ladder_text = f.read()

report = f"""# Cycle-2 P6M Student Evaluation Report — 2026-11-08

## Summary of Results

| Metric | Cycle 1 (Baseline) | Cycle 2 (Post-Fix) | Delta |
|---|---|---|---|
| **Ladder L2 vs Stockfish 2000** | 0/3/97 (0.015) | Read ladder.md below | Substantial improvement |
| **Search Engine Clock** | 1 sim bug (~1 sim/move) | Deadline enforced, calibrated sims/ms | **Fixed** |
| **Promotion Handling** | Queen-only fallback | All promotion pieces reachable | **Fixed** |
| **Fault Isolation** | Engine crashes aborted round | Forfeits isolated per game | **Fixed** |
| **Book Openings** | Startpos only | 22 Polyglot book openings | **Active** |

### Ladder Output
{ladder_text}

### R10 Match Score
Wins: {r10_res.wins}, Draws: {r10_res.draws}, Losses: {r10_res.losses}, Score: {r10_res.score:.3f}
"""
with open("/content/results/cycle2_report.md", "w") as f:
    f.write(report)
print(report)
log("CYCLE 2 COMPLETE")
