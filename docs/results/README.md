# Results

Reproduce a result from a clean checkout:

```bash
uv run python -m chessfoundation.train.pretrain --stage deep --data data/cache/deep --preset p20m --steps 100000
uv run python -m chessfoundation.eval.ladder --config ladder.toml
```

The ladder report records the pinned rung, the binary, engine options, time
control, opening book, tablebase path, achieved simulations, and Elo intervals
for every match. A run that does not clear L2 is reported as a negative result.

## Reports

- [2026-11-08 Cycle 2 Run (P6M, Colab L4)](2026-11-08-cycle2-p6m-colab-l4.md)
- [2026-11-07 Cycle 1 Run (P6M, Colab L4)](2026-11-07-p6m-colab-l4.md)
