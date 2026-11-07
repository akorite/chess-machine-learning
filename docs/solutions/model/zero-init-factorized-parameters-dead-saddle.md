---
title: Zero-initialized bilinear PV factors train to nothing
date: 2026-11-07
category: model
module: src/chessfoundation/model/heads.py
problem_type: logic_error
component: data_model
severity: high
symptoms:
  - "pv head loss stays constant across every training stage (policy/wdl move, pv does not)"
  - "pv accuracy metric never moves from chance"
  - "a parameter whose gradient is identically zero never warns — the run just wastes the auxiliary"
root_cause: logic_error
resolution_type: code_fix
tags: [initialization, auxiliary-head, gradient-dead-saddle]
---

# Zero-initialized bilinear PV factors train to nothing

## Problem
The PV auxiliary head used two factorized `nn.Parameter` tensors (`pv_left`, `pv_right`)
initialized to `torch.zeros`. Every pretraining and distillation stage reported a pv loss
that never moved — the head was dead from step 0.

## Symptoms
- `pv` logits identically zero at every checkpoint.
- Both factor parameters' gradients identically zero: the product `pv_left @ pv_right`
  derivative w.r.t. either factor contains the other factor, and both are zero, so no
  gradient flows into either.
- No error or warning; the auxiliary just wastes compute for the whole run.

## What Didn't Work
- More training steps — a dead saddle does not leave itself.
- QAT fake-quant and distillation rounds — they inherit whatever init the trunk has.

## Solution
```python
self.pv_left = nn.Parameter(torch.empty(pv_plies, dim, pv_rank).normal_(std=0.02))
self.pv_right = nn.Parameter(torch.empty(pv_plies, dim, pv_rank).normal_(std=0.02))
```

## Why This Works
Factorized/bilinear parameters only get gradients through the *other* factor
(∂/∂A of AB needs B ≠ 0). Initializing either factor to zero collapses both.
Small-random init gives each factor a nonzero signal; 0.02 keeps the product
near zero at init so the aux head starts weak rather than dominating the trunk.

## Prevention
- Any `nn.Parameter` consumed only inside a product/einsum with another learnable
  tensor needs nonzero init — check `grad` after one `backward()` in a smoke test,
  e.g. assert `param.grad is not None and param.grad.abs().sum() > 0` for every
  auxiliary-head parameter. Zero-init is safe only for output *bias*-style tensors
  feeding a linear/conv weight that is itself nonzero.
