# Experiment Log

## 2026-07-13 — successor scaffold and adversarial review

- Created as the parent LoRA pilot's preregistered full-rank capacity successor.
- Copied the self-contained task, recurrent mechanics, evaluation, and analysis
  harness; no parent source is imported at runtime.
- Replaced PEFT with 62 zero-initialized FP32 full-shape deltas on layers 12–19,
  active only for extra R calls.
- Added a strict parent-trigger reader, canonical parent-row parity contract,
  real Adam-state/memory G0, independent Carry/Bag K=1 call checks, and an
  observable delta-plus-loop checkpoint/logit round trip.
- Removed G4 from the CLI and verdict; it remains explicitly deferred.
- Ran CPU unit/static tests only. No data-preparation or model-bearing stage was
  run; there are no scientific results.

