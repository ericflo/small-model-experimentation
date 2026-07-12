# Qwen3.5-4B Native-Thought Jacobian Value Transport Report

## Status

Pre-run design and CPU data smoke. No model outcome is recorded.

## Purpose

Transfer the independently replicated context-local J mechanism from a
prompt-local lookup token to a natural native-thinking token, while separating a
task-general scalar continuation-value coordinate from full answer identity.

## Frozen method

- Fresh, first-operation-identifiable procedural list tasks only.
- Frozen replicated 24-token lens and layer band 4--8.
- Natural close, full-prefix batch-one Transformers recomputation without cache.
- Continuation-defined prefix value, held-out-by-task linear J-space readout,
  and a minimum-norm scalar coordinate clamp.
- Exact quantization-aware random controls and explicit identity, shuffled-axis,
  logit, raw, J/non-J, ActAdd, and wrong-donor arms.

## CPU evidence

All 80 generated tasks are fingerprint-unique and disjoint from the direct
Jacobian parent. Exhaustive enumeration certifies one visible-data-consistent
first-operation type. The CPU adversarial pass rejected `negate` as a target
because its two-step compositions are algebraically reorderable; it remains a
prompt distractor/second operation. This prevents alternate valid explanations
from being counted as model failures.

## Current inference boundary

No conclusion about native thinking, value decodability, or causal transport is
licensed. Model stages remain unavailable until the immutable design boundary is
committed and pushed.

## Oracle boundary

Continuation labels and high/low donor selection are hidden-label oracle inputs.
Even a terminal causal pass cannot support a capability claim or replace the
required learned non-oracle matched-sampling experiment.
