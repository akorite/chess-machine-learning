# chessfoundation

A small chess network distilled from public teachers, with Gumbel search and
bullet time management. The model is a hybrid ResNet stem plus smolgen
transformer (about 6M or 20M parameters) with policy, promotion, WDL, PV, and
moves-left heads.

## Install

```bash
uv pip install -e .
```

## Play

```bash
python -m chessfoundation.engine.model_engine \
  --checkpoint checkpoints/student.pt \
  --preset p20m --simulations 4096 \
  --book books/opening.bin --syzygy tablebases/
```

The engine speaks UCI; point any GUI at the command. Without `--checkpoint` it
plays from random weights, which is only useful for smoke tests.

## Data

```bash
python -m chessfoundation.data.build_cache --source lichess \
  --out data/cache/deep --count-only
python -m chessfoundation.data.build_cache --source lichess \
  --out data/cache/deep --shard-size 100000
python -m chessfoundation.data.build_cache --out data/cache/deep --verify
```

Depth discipline lives in one place: rows with `depth >= 24` or
`knodes >= 100000` feed the deep set; depth-10 data is pretraining only.

## Train

```bash
python -m chessfoundation.train.pretrain --stage value --data data/cache/gigafish --steps 20000 --out checkpoints/value.pt
python -m chessfoundation.train.pretrain --stage deep --data data/cache/deep \
  --init checkpoints/value.pt --steps 40000 --save-every 2000 \
  --resume checkpoints/deep.pt --out checkpoints/deep.pt
python -m chessfoundation.train.gates --checkpoint checkpoints/deep.pt --data data/cache/deep \
  --previous checkpoints/value-metrics.json
python -m chessfoundation.quant.verify --reference checkpoints/deep.pt \
  --candidate checkpoints/deep-int8.pt --data data/cache/deep --min-agreement 0.99
```

Pass `--accelerator L4 --hours 6 --phase pretrain` to any training command to run
the compute-budget preflight before the model loads; without `--accelerator` the
guard is skipped (local CPU runs).

A distillation round plays the student, labels the reached positions with a
teacher, and retrains on the round plus replay.

## Measure

```bash
python -m chessfoundation.eval.ladder --engine <student> --opponent <pinned stockfish> \
  --games 400 --movetime 50 --out report.md
```

The ladder stops when the score interval includes an even score and writes Elo
with 95% intervals. A result below the L2 floor is reported as a negative
result, not rounded up.

## Budget

`chessfoundation.budget` refuses a GPU run that would exceed its phase
allocation, the 150 CU ceiling, or the remaining balance minus a 5 CU reserve.
110 CU is the planning target.
