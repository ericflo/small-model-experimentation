# Qwen3.5-4B Jacobian Transport Control Replication Experiment Log

## Scaffold

Created as a new experiment scaffold.

## 2026-07-12 — intake and adversarial review

- Named the invalid 48/48 context-local clamp result as the direct parent.
- Froze its exact lens, band 4–8, alpha one, prompt grammar, and model revision.
- Registered fresh calibration/confirmation mappings and two independent random
  controls per item.
- Added post-bf16 gates for realized norm (1e-5 relative) and J-span projection
  fraction (0.01), plus wrong-donor same-span specificity.
- Completed the adversarial review before implementation or any model call.

## 2026-07-12 — immutable design boundary

- Pushed design commit `27b9da2a0973dbddbdfd2b6f7acddbfc7f4f736f`
  before model inference.
- Recorded exact frozen README/preregistration hashes and the byte-identical
  parent lens hash.

## 2026-07-12 — quantization-aware control implementation

- Copied the parent's cache-free batch-one Qwen patching and coordinate code,
  then added a numeric-only post-bf16 control optimizer.
- For 32 fixed random candidates per layer, the hook alternates realized-span
  removal/renormalization with 64-step scale search and chooses the first
  candidate meeting both frozen constraints. Candidate selection cannot access
  logits or labels.
- Implemented a model smoke and the 480-layer numeric calibration gate. The
  calibration writer rejects outcome-like fields and discards every forward's
  logits before serialization.
- CPU suite passes 24 tests plus 24 subtests; no model call has occurred in this
  replication.

## 2026-07-12 — model-smoke attempt 001

- Ran the first outcome-blind model smoke after pushing the implementation.
- Model/lens/token/position/causal contracts passed and 17/20 realized random
  layer deltas met both frozen constraints.
- Three layer-8 deltas had valid norm matching but realized J-span projection
  fractions 0.01116, 0.01214, and 0.01429, above the frozen 0.01 ceiling.
- Preserved the failed receipt as `runs/model_smoke/attempt_001_failed.json`.
  This is an engineering smoke failure, not a calibration or scientific result;
  calibration remains unopened and no outcome logits were recorded.
- The correction path had only retained scale-searched states at 16-iteration
  checkpoints. The implementation now retains each fixed candidate's lowest-
  projection intermediate quantization cell for a final geometry-only scale
  search, without changing seeds, candidate count, damping, iteration budget,
  binary-search budget, or either frozen threshold.

## 2026-07-12 — model-smoke attempt 002

- The retained-cell correction improved one failed layer-8 projection from
  0.01429 to 0.01135, but the smoke again passed only 17/20 rows; the other two
  failed projections remained 0.01116 and 0.01214.
- Preserved this outcome-blind receipt as
  `runs/model_smoke/attempt_002_failed.json`; calibration remains unopened.
- Audit found that scale search selected only the closest-norm state and could
  discard another visited bf16 plateau that better satisfies the joint frozen
  norm/projection gate. Scale search now ranks every visited state by the two
  preregistered constraints jointly. This repairs the implementation of the
  frozen gate; it changes no optimizer budget, seed, threshold, or outcome rule.

## 2026-07-12 — model-smoke attempt 003 and exact-lattice audit

- Joint scale selection exposed four layer-8 failures at projection fractions
  0.01037, 0.01054, 0.01135, and 0.01214; preserved the receipt as
  `runs/model_smoke/attempt_003_failed.json`.
- An outcome-blind diagnostic separated direction supply from live-lattice
  feasibility. On the unperturbed layer-8 state, 8,044--8,091 of 8,320 tested
  fixed-seed mixture states passed. After upstream random patches changed the
  live layer-8 bf16 state, zero mixture states passed; this localized the issue
  to sequential quantization, not favorable random-direction selection.
- For each failed live delta, exhaustively scored pairs of neighboring bf16
  coordinate moves using the exact norm and orthogonal-projector identities.
  One pair repaired three rows and two pairs repaired the fourth, producing
  projection 0.00852--0.00966 while keeping norm error <=9.02e-6.
- Integrated this deterministic pair repair only after the frozen continuous
  optimizer fails. It uses current residual, dictionary, target norm, and the
  same 512-step bound; it cannot inspect logits, labels, or answers and changes
  no seed, random draw, intervention, threshold, or endpoint.

## 2026-07-12 — model-smoke pass

- Fresh rerun with the exact-lattice implementation passed 20/20 numeric rows.
- Maximum post-bf16 relative norm error was 9.0113e-6 and maximum realized
  J-span projection fraction was 0.0098674.
- Four layer-8 rows required lattice repair: three used one pair and one used
  two pairs. All other rows passed without lattice moves.
- Exact model revision, lens rank/hash, token, position, equal-length, and
  causal-suffix contracts passed. The receipt records no outcomes.
- Calibration and confirmation remain unopened pending commit/push of this
  complete smoke history and adversarial implementation audit.

## 2026-07-12 — numeric calibration pass

- After the passing smoke boundary was committed, pushed, and green in CI, ran
  the frozen 24-item outcome-blind calibration once.
- All 480/480 rows passed. Maximum relative norm error was 9.8216e-6; maximum
  realized J-span projection fraction was 0.00999293; causal difference was 0.
- Thirty-seven rows used lattice repair: one each at layers 5, 6, and 7, plus
  34/96 at layer 8. Maximum repair was three coordinate pairs.
- Both prompt kinds and both random arms contain exactly 240 rows. No logits or
  outcome fields occur in the row artifact, and the summary records both as
  absent.
- Decision: `CONTROL_CALIBRATION_PASS`. Confirmation remains unopened pending a
  separate calibration commit/push.
