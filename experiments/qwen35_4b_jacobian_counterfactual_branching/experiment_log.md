# Qwen3.5-4B Jacobian Counterfactual Branching Experiment Log

## 2026-07-13 — Design and data preflight

- Completed idea intake, preregistration, and a 22-point adversarial review
  before model implementation or load.
- First proposed split seed `2026071301` was rejected before writing results:
  all 24 qualification fingerprints collided with a direct ancestor stream.
- Changed only the pre-outcome split seed to `2026072301`; collision checking
  remains fatal. No model or correctness metric had run.
- Regeneration produced 76 unique new fingerprints with zero overlap against
  634 direct-ancestor fingerprints: 4 mechanics, 24 qualification, 48 sealed
  confirmation.
- Four geometry tests and CPU smoke pass. Across layers 4--8 and alpha
  0.5/1/2, every J/non-J branch bank has width 12, rank 11, near-zero vector
  sum, Gram relative error at most 1.14e-6, and float non-J projection at most
  3.05e-7. Live bf16 controls remain for model smoke.
- Model stages remain fatal-unavailable pending cache-fork implementation,
  implementation audit, commit, and push.

## 2026-07-13 — Model-smoke implementation boundary

- Six implementation tests pass; pending-boundary invocation fails before model
  load.
- Pushed cache-free one-shot branch readout implementation at `9100395f` and
  froze exact runner/model/geometry/test hashes in config.
- This boundary authorizes outcome-blind model smoke only. It does not authorize
  alpha mechanics or continuation generation.
