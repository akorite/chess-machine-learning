# Shard-major streaming starves rare-label sources

**Symptom:** a 6M-parameter student trained on a 119.6M-record corpus scored 0/0/20 against a 1320-Elo rung, while its value metrics improved normally (`value_mae` 0.539 → 0.391).

**Cause:** the corpus is 84% value-only positions (GigaFish carries `fen` + `eval`, no move). Batches were streamed shard by shard to keep reads inside one mapping — correct for a corpus larger than page cache — but a 4,000-step run consumes only about four 1M-record shards. Those four can all be value-only, so the policy head trains on nothing and the loss silently reports `0.0` because the batch carried no labeled rows.

**Fix:** take a bounded number of batches per shard before advancing (`shard_batches=2`), so a short run visits every source many times while each batch still reads inside one file. A 32-batch window then spans ~16 shards instead of one.

**Diagnostic that separates this from undertraining:** play the student against an engine that plays the first legal move. A model with no policy signal still draws those games (it does not blunder) but never converts; a merely undertrained model wins some. Also watch the policy loss term across checkpoints: `0.0` means unlabeled batches, not a good policy.

**General rule:** any shard-ordered sampler over a multi-source corpus needs a source-aware visit schedule, or the rarest label type gets sampled at the rate of its shard count rather than its share of records.
