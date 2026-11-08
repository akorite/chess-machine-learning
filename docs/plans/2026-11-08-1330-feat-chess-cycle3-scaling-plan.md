---
artifact_contract: ce-unified-plan/v1
product_contract_source: ce-plan-bootstrap
execution: code
created: 2026-11-08
origin: docs/plans/2026-11-07-1035-feat-chess-foundation-model-plan.md
---

# feat: Cycle 3 — scale the student past the 100M-position regime

## Goal Capsule

- **Objective:** The shipped student plays at a level far above cycle 2's ~1350 Elo plateau — measurable as a ladder score against a pinned Stockfish rung that the cycle-2 checkpoint could not reach — and the pipeline that produced it is reproducible from one command per stage.
- **Means:** A >100M-position training corpus, a shuffled prefetching training loop, and four representational fixes to the existing P6M family (side-to-move canonicalisation, per-square positional information, continuous value targets, decoupled PV head) (KTD26, KTD27, KTD28, KTD29).
- **Authority hierarchy:** This plan > origin plan Product Contract R1–R13 (preserved) > repo conventions. Where this plan is silent, the origin plan governs.
- **Stop conditions:** Colab balance would drop below 25 CU; a 100-position cache build cannot complete inside the per-phase budget; the ladder cannot play a single legal game.
- **Execution profile:** one branch, one PR.
- **Finisher:** `ce-work` in return-to-caller mode, then the lfg pipeline owns simplify, review, and delivery.

## Product Contract

### Summary

Cycle 2 fixed the measurement and reliability defects (engine clock, oracle faults, rung protocol, book, CU ledger) but shipped a student trained on 1.8M positions that lost every game against Stockfish 2000. Cycle 3 attacks training scale and the four representational bottlenecks that make scale buy less strength than it should: absolute (not side-to-move) board orientation, position-blind attention, bucketed value targets, and a PV head entangled with the policy projection.

### Problem Frame

The cycle-2 result was 0/2/48 against Stockfish `UCI_Elo 2000`, score 0.020. The engine's mechanics are now correct — wall-clock deadlines hold, search no longer picks unsearched blunders, fault isolation works — so the remaining gap is the learned function, not the machinery. Three facts bound the diagnosis:

1. **Training volume.** The evaluated checkpoint saw ~1.8M positions in 7k steps. Published comparable work uses 10M games / 15B annotated points (ChessBench) or billions of self-play positions.
2. **Representation.** The model sees an absolute board (White at rank 1) while its value target is side-to-move; it must learn every pattern twice. Its 64 square tokens carry no positional embedding, and `smolgen` averages all squares into one vector before generating attention bias, so attention cannot key on square identity.
3. **Supervision quality.** Value targets are three buckets split at ±50 cp, so +0.55 and +9.00 are the same label, and the 6-ply PV head shares `src`/`dst` projections with the policy head while carrying weight 0.5 — cycle 1 recorded `pv` loss pinned near `ln(4096)`, i.e. it was injecting noise straight into the policy gradient.

Cycle 2 also measured the loop at ~1.93k positions/s (sequential, unshuffled `batch_slice`), which caps what any position budget can buy in a session.

### Requirements

- **R14 — Corpus scale.** The training corpus exceeds 100M labeled positions in total, mixed from at least two of: Lichess position evaluations (depth ≥ 20 filter), GigaFish depth-10 evaluations, ChessBench action-value shards. The cache manifest records per-source record counts and the depth bands present.
- **R15 — Throughput.** The training loop consumes shuffled batches at ≥ 6,000 positions/s on a Colab L4 at batch size ≥ 512, measured by the loop itself and recorded with the run. Cycle-2 baseline is 1.93k positions/s.
- **R16 — Colour equivariance.** A position and its colour-mirrored twin produce identical policy distributions (after mirroring the move indices back) and identical value estimates.
- **R17 — Square identity.** Square tokens carry learnable positional information, and the attention-bias generator preserves per-square structure rather than pooling all 64 tokens into one vector.
- **R18 — Continuous value.** Value supervision is a continuous win-probability target derived from the recorded centipawn score, not a hard ±50 cp three-way bucket. WDL classification stays as an auxiliary head.
- **R19 — Head separation.** PV supervision uses projections independent of the policy head, and its loss weight is low enough that PV error cannot dominate the policy gradient.
- **R20 — Reproducible training.** One command trains a stage with shuffle, warmup+cosine schedule, periodic validation metrics, and checkpoint save/resume whose worst-case lost work is ≤15 minutes of training.
- **R21 — Evaluation.** The trained checkpoint is evaluated under the pinned 1+0 protocol with the opening book, with achieved simulations read from PGN telemetry, against the ladder rungs and one R10-class opponent.
- **R22 — Budget.** Total Colab compute for this cycle stays inside the remaining balance, is recorded in the cumulative ledger, and is reported against the starting balance.

### Key Decisions

- **Scale is bought with data volume first, capacity second.** <!-- Governs R14, R15 -->
  Reason: the cycle-2 model is undertrained by three orders of magnitude, and data volume is the axis with published monotone returns. Widening the preset is considered only after the data pipeline can feed it.
- **Architecture fixes ship before the long training run.** <!-- Governs R16, R17, R18, R19 -->
  Reason: R16 doubles effective sample efficiency and costs a few lines; discovering it after a 100M-position run would require repeating the run.
- **Compute budget is authorised up to the remaining Colab balance.** (session-settled: user-directed — chosen over keeping spend minimal as in cycle 2's 1.4 CU: the user authorised spending the remaining budget to reach a working engine.)
- **A faster, more expensive GPU is acceptable when it reduces wall-clock.** (session-settled: user-directed — chosen over staying on L4 at 1.94 CU/hr: the user accepted a higher CU/hour rate in exchange for shorter runs.)
- **The training loop is rewritten for throughput rather than tuned.** (session-settled: user-directed — chosen over reusing the sequential `batch_slice` sampler: the user asked for a better training loop, and 1.93k pos/s cannot consume 100M positions in a session.)
- **The corpus target is >100M positions.** (session-settled: user-directed — chosen over the cycle-2 ~2M-position corpus: the user explicitly authorised going past 100M.)

### Success Criteria

- Ladder score against the highest cleared rung, with a 95% interval that excludes 0.5, or an honest negative result naming the highest rung beaten and the loss attribution.
- `docs/results/` gains a cycle-3 report stating corpus size, per-source counts, measured positions/s, achieved simulations per move, and CU spend.
- Reproducing the run needs only the documented commands plus the recorded seed.

### Scope Boundaries

In scope: data pipeline, training loop, model representation, value supervision, head separation, corpus build, training run, evaluation, report.

Out of scope (deferred or excluded):

- A 100M+-parameter model. The match-preset latency budget still governs the engine, and cycle 3 buys strength from data before capacity.
- Real INT8/TensorRT export (unchanged from the origin plan's KTD17).
- Self-play RL from scratch. Distillation exists (U15) but the cycle-3 gain is supervised scale.
- Re-training the Lc0 teacher or re-deriving AVI targets.

### Risks & Dependencies

- **Data-build wall clock.** Streaming 100M rows through `chess.Board` parsing is slow; PV parsing in `record_from_eval` is the dominant cost and must be reduced (KTD31). Mitigation: parse only the principal move, parallelise across shards.
- **`RECORD_DTYPE` change invalidates every existing cache.** Cache files are regenerated, not migrated; the cycle-2 caches are discarded.
- **Long-run interruption.** Colab VMs are ephemeral. Mitigation: 15-minute checkpoint cadence plus `--resume`, and the run writes its metrics file per checkpoint.
- **Colour-flip bugs are silent.** A wrong mirror produces a model that quietly plays badly. Mitigation: R16 is proven by a test comparing both orientations, not by inspection.

### Sources

- `docs/plans/2026-11-07-1035-feat-chess-foundation-model-plan.md` (origin Product Contract R1–R13, KTD1–KTD25).
- `docs/results/2026-11-08-cycle2-p6m-colab-l4.md` (cycle-2 measurements this plan responds to).
- Lc0 "Transformer Progress" (BT3/BT4 architecture: per-square embedding projection, smolgen, small head depth, weak FFN scaling) — https://lczero.org/blog/2024/02/transformer-progress/
- Ruoss et al., *Amortized Planning with Large-Scale Transformers* (NeurIPS 2024): ChessBench = 10M games / 15B annotated points from Stockfish 16; action-value targets; dataset size is the dominant ablation.
- Danihelka et al., *Policy Improvement by Planning with Gumbel* (ICLR 2022) — already implemented in cycle 2.

## Planning Contract

### Key Technical Decisions

- **KTD26 — Side-to-move canonicalisation.** The network sees the board from the side to move: when Black moves, piece planes swap colour channels and ranks mirror; policy/PV outputs mirror back to absolute squares on the way out. This halves the pattern space the model must learn and makes R16 testable by construction. Legal move masks, tablebase, and book paths stay in absolute coordinates.
- **KTD27 — Per-square positional information.** A learnable `(1, 64, d_model)` positional embedding is added to the token sequence, and smolgen's generator consumes the per-square projected tokens (`(B, 64, gen)` flattened) instead of `x.mean(dim=1)`, so the attention bias can depend on which piece stands where (Lc0 BT3+). Head depth stays 8 or below per the published finding that small head depth suits 64-token inputs.
- **KTD28 — Continuous value target.** The record carries the centipawn score; the training target is `win_prob = 1 / (1 + 10^(-cp/400))` in side-to-move perspective, regressed with MSE, while the existing three-class WDL head stays as an auxiliary classifier. Mate scores clamp to ±1. This replaces bucketing as the primary value signal without discarding the calibrated WDL head the engine's search consumes.
- **KTD29 — PV head separation.** The PV head gets its own `pv_src`/`pv_dst` projections and the loss weight drops to 0.05, so a PV regression can no longer corrupt the policy projection (cycle-1 evidence: `pv` loss stuck at 8.313 ≈ ln(4096) while weighted 0.5).
- **KTD30 — Shuffled shard sampler.** Training index order is a deterministic permutation over the concatenated shard lengths, reshuffled per epoch from a seeded generator, with batch tensors materialised in worker processes and pinned memory feeding the device. Deterministic given the seed, so a resumed run reproduces the same order.
- **KTD31b — Row-group parallelism for the corpus build.** A parquet shard yields several jobs, one per contiguous run of row groups, each with its slice of the source's row budget. A 57M-row evaluation file converted by one worker made every other shard wait behind it; row groups are independently readable, so the build scales to all workers instead of one.
- **KTD31 — Cheap record construction.** `record_from_eval` parses only the principal move from the PV for the policy target and skips full-line replay; corpus builds fan out across shards with a process pool. This is what makes a 100M-row build fit the budget.
- **KTD32 — Source mix, not a single source.** GigaFish supplies value volume at depth 10, Lichess supplies deep policy/PV, ChessBench supplies Stockfish-16 action values with game outcomes; the mix is a CLI argument so the ratio can be re-run without code change.
- **KTD33 — Preset stays P6M for the shipped artifact.** Data and representation carry this cycle; a wider preset is a later cycle's axis (origin KTD14 still governs the match-preset latency budget). If throughput headroom proves large, a `p20m` comparison run may be recorded as evidence, not shipped.

### High-Level Technical Design

```mermaid
flowchart LR
  A[parquet sources<br/>lichess · gigafish · chessbench] --> B[build_cache --mix<br/>pool per shard]
  B --> C[shards + manifest<br/>counts per source/band]
  C --> D[ShuffleSampler<br/>seeded permutation]
  D --> E[worker batch assembly<br/>canonicalise + planes]
  E --> F[student<br/>pos-embed + smolgen]
  F --> G[policy · promo · pv<br/>value(reg) · wdl(aux)]
  G --> H[loss<br/>CE + MSE + KL]
  H --> I[checkpoint + metrics.json<br/>every N steps]
  I --> J[ladder under 1+0 + book]
```

Canonicalisation sits between record and model, and its inverse sits between model output and move choice. Both directions live in `data/encode.py` so training, search, and the colour-flip test use one implementation.

### Assumptions

- Colab still grants L4 and A100 allocations for this account; measured balance at plan time is 247.10 CU.
- Lichess position evaluations, GigaFish and ChessBench remain downloadable at their current repo paths.
- A 100M-row cache fits the Colab disk (measured: 179 GB free on the cycle-2 VM; a record is ~80 bytes, so 100M rows ≈ 8 GB).

## Implementation Units

### U18. Streaming shuffled sampler and batched loader

- **Goal:** Training consumes shuffled batches at ≥ 6,000 positions/s on L4 with deterministic order from a seed.
- **Requirements:** R15, R20
- **Dependencies:** none
- **Files:** `src/chessfoundation/data/cache.py`, `src/chessfoundation/train/loop.py`, `src/chessfoundation/train/throughput.py`, `tests/test_train.py`, `tests/test_cache.py`
- **Approach:** Add a seeded permutation index over the concatenated shard lengths and a batched reader that materialises a whole batch per call; keep `ShardDataset.__getitem__` for single-record callers. Replace `batch_slice`'s sequential window in `loop.train` with the sampler; add `num_workers`/`prefetch` awareness only where it measurably helps (worker process batch assembly behind a bounded queue). `throughput.measure` reports positions/s at the plan's batch size.
- **Test scenarios:** Two samplers with the same seed yield identical index order across epochs; different seeds differ; the sampler covers every index once per epoch when the batch size divides the dataset; a resume at step N yields the same batch order as an uninterrupted run; throughput reports ≥6,000 pos/s on the reference config or records the achieved number.
- **Execution note:** Measure before and after with `train.throughput`; the red signal is the pre-change positions/s.

### U19. Record carries centipawn score

- **Goal:** Every record can produce a continuous value target.
- **Requirements:** R18
- **Dependencies:** none
- **Files:** `src/chessfoundation/data/encode.py`, `src/chessfoundation/data/build_cache.py`, `tests/test_encode.py`
- **Approach:** Add a signed `cp` field to `RECORD_DTYPE` with a sentinel for unknown, populate it in `record_from_eval` and the ChessBench path (converting the action value back to centipawns), and expose `win_prob(record)` returning the side-to-move win probability. Old caches do not load; document the rebuild.
- **Test scenarios:** `win_prob` returns 0.5 at cp=0, >0.9 at cp≥+400 from the mover's perspective, and mirrors sign for Black to move; mate scores clamp to 0/1; a ChessBench row yields the expected cp.
- **Execution note:** Proof-first — the dtype change breaks existing cache reads, so the encode test is the red signal.

### U20. Side-to-move canonicalisation

- **Goal:** The model is colour-blind by construction.
- **Requirements:** R16
- **Dependencies:** U19
- **Files:** `src/chessfoundation/data/encode.py`, `src/chessfoundation/engine/gumbel.py`, `src/chessfoundation/engine/model_engine.py`, `src/chessfoundation/train/loop.py`, `tests/test_encode.py`, `tests/test_engine.py`
- **Approach:** `canonical_planes(planes, black_to_move)` swaps colour channel blocks and vertical-flips square planes; `canonical_move_index` maps `from*64+to` through the mirror; `uncanonical_move_index` returns it. Training feeds canonical planes; policy targets are canonicalised; search maps its chosen index back before minting the `chess.Move`. Promo indices are unaffected.
- **Test scenarios:** A position and its colour mirror produce identical canonical planes; canonicalising twice is identity; move-index mirror round-trips for all 4096 indices; the engine returns a legal absolute move for a Black-to-move position; a fixed-seed search yields the mirrored move after mirroring the board.
- **Execution note:** The equivariance test is the acceptance proof: both orientations must produce equal outputs.

### U21. Positional embedding and per-square smolgen

- **Goal:** Attention can distinguish squares and the bias generator keeps per-square structure.
- **Requirements:** R17
- **Dependencies:** none
- **Files:** `src/chessfoundation/model/student.py`, `src/chessfoundation/model/smolgen.py`, `src/chessfoundation/model/presets.py`, `tests/test_model.py`
- **Approach:** Add `pos_embed` to `Student.encode`; change smolgen's generator input from the pooled mean to the flattened per-square projection with a matching reduction, keeping parameter growth small. Head depth stays ≤8 in presets.
- **Test scenarios:** Two boards that differ only by moving one piece produce different attention bias tensors; permuting the token order changes the output (proves positional information is used); a fixed input is invariant to batch composition; parameter count growth is bounded and reported.
- **Execution note:** The permutation test is the red signal — pre-change, permuting tokens leaves outputs unchanged.

### U22. Continuous value head and decoupled PV head

- **Goal:** Value supervision is continuous and PV error cannot poison policy.
- **Requirements:** R18, R19
- **Dependencies:** U19, U21
- **Files:** `src/chessfoundation/model/heads.py`, `src/chessfoundation/train/loop.py`, `tests/test_model.py`, `tests/test_train.py`
- **Approach:** Add a `value` scalar output to `Heads`; keep `wdl` as the auxiliary classifier; give PV its own projection parameters; set loss weights `value=1.0`, `wdl=0.3`, `pv=0.05` and record each component in metrics. The engine's search keeps consuming the WDL softmax, so latency and search code are untouched.
- **Test scenarios:** `compute_loss` returns a finite `value` term and a `pv` term that does not change when the policy projection is perturbed (separation proof); a record whose cp moves from +30 to +600 raises the target value; the WDL head still produces a 3-vector consumed by search.
- **Execution note:** Proof-first on the separation test: perturbing `src` must not move `pv` gradients.

### U23. Mixed-corpus builder with parallel shard workers

- **Goal:** A >100M-position corpus builds inside the budget with per-source counts.
- **Requirements:** R14, R22
- **Dependencies:** U19, U20
- **Files:** `src/chessfoundation/data/build_cache.py`, `src/chessfoundation/data/sources.py`, `tests/test_build_cache.py`
- **Approach:** Add `--mix` accepting `source:count` pairs that concatenates the per-source record streams and caps each; fan shard processing across a process pool; parse only the principal PV move (KTD31); record per-source and depth-band counts in the manifest as records land.
- **Test scenarios:** `--mix` caps each source and the manifest counts sum to the total; a mix of two local fixture parquets yields records from both; parallel build output equals the serial build for the same inputs (same record multiset); an interrupted build leaves a manifest whose shards verify.
- **Execution note:** The interrupted-build test proves the resumable prefix still holds after the pool change.

### U24. Train command with validation and resumable checkpoints

- **Goal:** One command trains a stage with metrics and ≤15-minute checkpoint loss.
- **Requirements:** R20, R15
- **Dependencies:** U18, U22
- **Files:** `src/chessfoundation/train/pretrain.py`, `src/chessfoundation/train/loop.py`, `tests/test_pretrain.py`
- **Approach:** Add periodic validation (held-out slice: value MAE, policy top-1, WDL accuracy) written to `<out>.metrics.json` alongside each checkpoint; default `--save-every` derived from measured step rate so worst-case loss is ≤15 minutes; keep `--init` (weights only) and `--resume`.
- **Test scenarios:** A 5-step run writes a metrics file containing the validation keys; `--save-every 2` over 5 steps writes checkpoints at 2 and 4; resuming from a mid-run checkpoint continues at the recorded step and reproduces the same metric keys.
- **Execution note:** Keep the run short in tests (2-5 steps on fixture shards).

### U25. Corpus build and scaled training run

- **Goal:** The corpus exceeds 100M positions and a checkpoint is trained on it inside the budget.
- **Requirements:** R14, R15, R20, R22
- **Dependencies:** U23, U24
- **Files:** `scripts/colab_cycle3_execution.py` (new), `docs/results/2026-11-08-cycle3-p6m-colab.md` (new)
- **Approach:** Provision a Colab GPU, stage the repo, build the mixed corpus, train with the measured throughput, checkpoint to a predictable path, download the checkpoint and metrics.
- **Test scenarios:** Report records corpus size per source, measured pos/s, steps, and final metric components.
- **Execution note:** This unit's completion signal is out-of-repo (the VM run); it is complete only when the checkpoint and metrics exist locally.

### U26. Ladder evaluation and cycle-3 report

- **Goal:** A defensible strength number, or an honest negative result with attribution.
- **Requirements:** R21, R22
- **Dependencies:** U25
- **Files:** `docs/results/2026-11-08-cycle3-p6m-colab.md`, `docs/results/README.md`
- **Approach:** Run the pinned ladder under 1+0 with the book against Stockfish `UCI_Elo` rungs, then one R10-class opponent; read achieved simulations from PGN telemetry; report against the cycle-2 baseline and the CU ledger.
- **Test scenarios:** Report states corpus size, pos/s, achieved sims, per-match protocol, intervals, and delta vs cycle 2.
- **Execution note:** A sub-L2 outcome is reported as a negative result naming the highest rung beaten.

## Verification Contract

| Check | Command or artifact | Exit criterion |
|---|---|---|
| Unit tests | `uv run pytest -q` | all pass |
| Lint | `uv run ruff check src/ tests/` | clean |
| Throughput | `uv run python -m chessfoundation.train.throughput --preset p6m --accelerator L4` | ≥6,000 pos/s at batch ≥512, or the achieved number recorded |
| Colour equivariance | `uv run pytest tests/test_encode.py -k canonical` | mirrored boards produce equal outputs |
| Square information | `uv run pytest tests/test_model.py -k permut` | token permutation changes output |
| Head separation | `uv run pytest tests/test_train.py -k pv` | policy perturbation leaves PV term unchanged |
| Cache integrity | `uv run python -m chessfoundation.data.build_cache --verify --out <dir>` | checksums match, per-source counts reported |
| Corpus size | manifest `counts.records` | > 100,000,000 |
| Training run | metrics file from U25 | finite losses, validation metrics present |
| Ladder | `uv run python -m chessfoundation.eval.ladder --engine <uci> --opponent <uci> --book data/book.bin` | score interval reported; rung claimed only when it excludes 0.5 |
| Budget | `colab usage` before/after | ledger total ≤ remaining balance, ≥25 CU reserve |

## Definition of Done

- R14–R22 satisfied with the evidence each names.
- Corpus manifest verifies and reports >100M records with per-source counts.
- The trained checkpoint is documented with its corpus, steps, measured throughput, and validation metrics.
- The cycle-3 results report exists under `docs/results/` and is linked from `docs/results/README.md`.
- Dead-end and experimental code from approaches that did not pan out is removed; no throwaway scripts remain committed except the reproducible runner.
- Tests and lint pass on the final tree.
