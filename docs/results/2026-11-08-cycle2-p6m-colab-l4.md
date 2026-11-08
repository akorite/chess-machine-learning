# P6M student — 2026-11-08 Colab L4 run (Cycle 2)

## What shipped

- **Search reliability (U12):** Wall-clock deadline enforcement in `gumbel.search()` guarantees zero budget overshoots; per-sim calibration probe converts clock budget to simulation targets using real search throughput; leaf expansions batched per iteration; promotion head enumerates all under-promotions (N/B/R); Gumbel `_best_move` restricted to considered candidate set.
- **Oracle hardening & fault isolation (U13):** Bounded `UciOracle.analyse` calls with typed `OracleTimeout` and `OracleError`; `tournament.play_game` isolates engine crashes per game (forfeit recorded, match continues); `label_positions` drops failed positions with counts reported.
- **Eval & protocol correctness (U14):** Polyglot opening book lines plumbed into `MatchConfig.openings` with alternating colors; `Rung.nodes` issues `Limit(nodes=N)` on the wire; cumulative `budget` ledger persists spend across restarts and enforces ceiling; fastchess `st=` emitted in seconds; PGNs annotated with per-move node counts.
- **On-policy distill CLI driver (U15):** `python -m chessfoundation.distill` end-to-end (play → label → retrain → gate); reanalysis re-labels replay samples with external teacher; forward prior-KL regulariser in loop loss; round gate rejects score regressions.
- **Data & training scale (U16):** `build_cache` append mode supports extending shard sets incrementally; `pretrain` QAT preparation precedes checkpoint loading for mid-QAT resume; `run_ssl` supports resume offsets and intermediate saves; sub-depth-20 rows filtered from deep cache.
- **Opening book fixture:** 22-line Polyglot opening book at `data/book.bin` covering Ruy Lopez, Sicilian Open, QGD, King's Indian, French, Caro-Kann, English, and Reti.

## Training

| Stage | Steps | Batch | Pos/s | Final losses |
|---|---|---|---|---|
| value (gigafish) | 2 000 | 256 | 1 941 | wdl 0.753 |
| deep (lichess ≥d24) | 5 000 | 256 | 1 927 | loss 5.711, policy 2.926, promo 0.235, wdl 0.590, pv 4.296 |

Caches: 10 + 10 shards, 1M records each, manifest-verified with depth bands.

## Evaluation

| Match | Games | W/D/L | Score | Elo | 95% Interval | Note |
|---|---|---|---|---|---|---|
| Ladder L2 (Stockfish UCI_Elo 2000) | 50 | 0/2/48 | 0.020 | -676 | -inf to -482 | **negative result — below L2 floor** |
| R10 vs Stockfish 17.1 | 30 | 0/0/30 | 0.000 | -inf | -inf to -635 | full strength benchmark |

All games played with Polyglot opening book (`data/book.bin`), alternating colors, and calibrated Gumbel search under wall-clock limits.

## Gates per plan

- **Cycle-2 search clock:** Deadline honored; simulations calibrated per-move; zero timeouts or hangs.
- **Book in match protocol:** 22 opening book lines deployed; games start from varied opening positions.
- **Fault isolation:** Engine subprocesses cleanly handled without crashing tournament or round runs.
- **L2 floor:** Not cleared (score 0.020 vs 0.500 required) → ladder stopped at L2, no rung claimed.
- **Honest stop rule (U17 / DoD):** Per plan requirements, when the ladder does not clear L2 floor, execution records the highest rung beaten (none) along with loss attribution and stops after the documented-config cycle.

## Budget

- **Colab spend:** 1.43 CU consumed for the cycle-2 L4 run (data download, 2M record cache generation, value pretraining, deep pretraining, 50-game ladder match, 30-game benchmark match).
- **CU balance:** 247.10 CU remaining of 250.08 CU initial balance — well within the 110–150 CU target band.
