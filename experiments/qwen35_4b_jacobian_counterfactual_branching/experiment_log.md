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

## 2026-07-13 — Invalid first live-bf16 receipt

- The first outcome-blind model smoke loaded the exact model/lens and applied
  every hook once, but its numeric receipt was invalid.
- Root cause: `current` was a view into the cloned activation; assigning the
  changed activation mutated the supposed before-state, so all realized deltas
  were falsely recorded as zero.
- A second guard omission reported J requested-norm error 1.0 but did not include
  it in the conjunctive numeric pass, producing an impossible false pass.
- No branch probabilities, choices, correct alias, outcome, qualification, or
  confirmation data were recorded. The receipt is preserved as
  `model_001_invalid_receipt.json` and cannot authorize mechanics.
- Fixed by cloning the float before-state before assignment, computing realized
  deltas against it, and making J requested-norm error a mandatory gate. A new
  pushed implementation hash is required before rerun.
