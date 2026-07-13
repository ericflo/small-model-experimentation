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
- Corrected implementation was pushed at `802cf1a5`; config now anchors its
  updated runner/model/test hashes. Smoke 002 is authorized only after this
  re-anchor commit is pushed.

## 2026-07-13 — Valid smoke 002 requires quantization-aware control repair

- Corrected deltas are nonzero and the receipt is valid/outcome-blind.
- Naïve bf16 non-J controls missed paired J norm by up to 3.39% and leaked up to
  2.96% into J, failing the frozen 1e-5/1% live gates.
- Requested J fidelity (4.16%), realized Gram (5.83%), and zero-sum residue
  (0.015625) remain diagnostics; exact Gram/rank/zero-sum is the pre-bf16 CPU
  construction gate, while paired norm/span are the explicitly frozen live
  gates.
- Added vectorized outcome-blind repair of each fixed non-J request toward its
  paired realized J norm outside the complete J span, with 512 iterations and
  0.5 damping. Code must be pushed/re-anchored before smoke 003.
- Repair implementation pushed at `934f4d59`; exact updated hashes are anchored
  for smoke 003.

## 2026-07-13 — Smoke 003 reaches 51/60, exact lattice repair next

- Iterative outcome-blind repair validates all 12 rows at layers 5--7 and
  reaches the paired norm/span boundary on layer 4; layer 8 retains nine fails.
- Maximum remaining norm error is 1.3206e-5 and span leakage 2.096%; no tolerance
  changed.
- Added the independent transport replication's exact one-ULP pair search for
  only failing rows, bounded at 32 pairs. Code must be pushed/re-anchored before
  smoke 004.
- Lattice implementation pushed at `dc298278`; exact runner/model/test hashes
  are anchored for smoke 004.

## 2026-07-13 — Smoke 004 fixes layer 8, misses one layer-4 receipt boundary

- Exact lattice search validates all layer-8 rows with at most five pairs; all
  independent span projections pass.
- Independent layer-4 paired norm is 1.033804e-5 versus 1e-5, although the
  device-local repair check stopped inside the boundary.
- Added a stricter 0.95 geometry repair objective so near-boundary rows cannot
  stop until they have 5% guard band. Scientific tolerances remain unchanged.
- Code must be pushed/re-anchored before smoke 005.
- Safety-margin implementation pushed at `6660bd94`; exact hashes are anchored
  for smoke 005.
