# State-Carry Versus State-Bag Experiment Log

## 2026-07-12 — Intake and Design Freeze

- Attached the new experiment to `structured_execution_and_compilers`; `test_time_reasoning_budget` was rejected as primary because its charter excludes new architectures.
- Named `qwen_fastweight_hook` as the closest negative near-duplicate.
- Reconstructed repository evidence C11–C54 and current recurrence/latent-state literature as a failure map.
- Froze the central contrast: one inherited state versus equal-compute independent reset states.
- Froze Qwen layers 12–19 as two complete hybrid motifs, eight state slots, K=4 training, K=5–12 extrapolation, three seeds, and fail-closed verdicts.

No model was loaded or called.

## 2026-07-12 — Implementation

- Removed the scaffold's vLLM runner because hidden-state intervention requires Transformers and backend mixing would invalidate comparisons.
- Added deterministic random-world substrate, three transition families, three renderings, exact trajectories, structural fingerprints, held-out depth/family/template splits, and matched counterfactual pairs.
- Added a manual pinned-Qwen forward with untouched K=1, recurrence-only loop LoRA, state-only cross-loop communication, Carry/Bag edge switch, state sufficiency heads, fixed-point loss, and optional semantic echo.
- Added model identity/layer/tokenizer/LoRA-locality/parity/gradient gates.
- Added paired pilot/full training, matched-depth evaluation, K curves, edge cuts, donor swaps, explicit textual trace training, compute-matched sample-more, paired bootstrap, and terminal verdict assignment.
- Hardened minimum-depth generation after adversarial review: both full joint-state repeats and earlier occurrences of the terminal queried field are rejected, including in donor-swap pairs.
- Added research handoff, literature map, architecture contract, GPU runbook, agent goal, preregistration, and adversarial design review.

## 2026-07-12 — Local Validation

- CPU smoke: `CPU_SMOKE_PASS`; three families at depths 1/4/8; counterfactual pair has distinct consequences; Carry/Bag compute receipts identical; no benchmark files read.
- Deterministic smoke data build: zero structural cross-split duplicates; exact row counts and hashes.
- Unit suite: 25 tests pass after initial implementation and adversarial hardening.
- Python compilation: every experiment source and script compiles without importing unavailable GPU packages.

No Qwen model was loaded or called. Live model smoke remains the first task on the 48 GiB Ada environment.

## Review Revisions

- Avoided duplicate registration of the PEFT base model inside the recurrence wrapper.
- Removed latent workspace placeholders from the explicit-text comparator.
- Made answer tokens context-prefix-stable rather than assuming standalone tokenization.
- Corrected the explicit-CoT target to close Qwen's think channel before the final answer.
- Isolated analysis by config hash so mixed echo cannot pool with continuous results.
- Reallocated evaluation compute toward full matched-depth and K=4 comparisons; nonprimary K curves are smaller diagnostics.
- Preserved both composite and text-only Qwen3.5 config identifiers while keeping model ID/revision absolute.
- Strengthened counterfactual swaps so paired prompts share world, label mapping, table order, query, and choice order; only the initial state and consequence differ.
- Replaced flat item bootstraps with hierarchical seed-then-task resampling, machine-enforced positive breadth on six of eight depths, and an explicit state-sufficiency verdict gate.
- Made sample-more fail closed on compute overspend and reject truncated thoughts that never naturally reach the answer channel.
- Corrected auxiliary node supervision from an unobservable random generator ID to the node's visible table-row position; counterfactual pairs share that coordinate system.
- Made gzip archives byte-reproducible and removed the counterfactual exception from the global structural-duplicate firewall; the complete default-size corpus now builds cleanly.
- Fixed portable row receipts, pilot/full isolation, primary-cell completeness enforcement, and edge-cut analysis; corrupted rows, datasets, adapters, and loop states now fail hash checks.
- Added initial-value and cumulative training-compute receipts that analysis enforces for every Carry/Bag seed pair.
- Matched the explicit-CoT optimizer schedule and upgraded deployment analysis to a three-seed task-paired hierarchical comparison against oracle `pass@N`, with actual sampled-token and synchronized timing receipts.

## 2026-07-12 — Final Adversarial Pre-Run Revision

No model was loaded or called. Three independent read-only reviews covered scientific design,
implementation, statistics, artifacts, and GPU operations; the primary agent then re-read every
experiment file and traced config→data→model→training→evaluation→analysis.

- Fixed cross-process corpus nondeterminism from set iteration and added a multi-`PYTHONHASHSEED` regression.
- Moved G1 to seed 7401 and dedicated pilot-only depth, joint, and counterfactual splits; confirmation keeps seeds 7411–7413 and all scored rows untouched.
- Replaced the nested seed/task bootstrap with a crossed bootstrap over the common task×training-seed matrix and added strict duplicate/key/corpus checks.
- Made exact checkpoint phase, fixed-final step, seed, critical-source digest, environment-lock digest, tensor identity, and ordered training-row digest mandatory.
- Converted the same-checkpoint edge cut from artifact availability into a positive causal gate with complete cells, per-seed direction, and crossed uncertainty.
- Balanced and retained query type, made joint state accuracy mandatory, and added a joint family+surface holdout gate.
- Geometry-matched counterfactual pairs at a shared initial node, evaluated both swap directions, hashed raw interventions, and added pre/post donor-following evidence.
- Hardened the explicit-CoT comparator with frozen sampling/allocation, raw generations, exact full coverage, compute rechecks, and close/parse/cap plus Carry answer-mode gates.
- Removed the conditional mixed semantic-echo variant; any interface follow-up now requires a separate experiment and its missing shuffled/wrong-task controls.
- Declared training non-resumable rather than allowing approximate recovery, reduced checkpoints to fixed finals, made shell loops fail-fast, and added a worst-format K=12 G0/resource receipt.

The prior CPU-smoke receipt predates these changes and is historical only. Unit/static validation is
rerun after the patch; a fresh CPU smoke/data manifest remains the first operational step before G0.

The user additionally required the low-rank capacity ambiguity to be resolved rather than left as a
caveat. Rank-32 LoRA remains first. A valid miss that fails to establish deep state formation mandates
creating and executing a new zero-initialized full-rank extra-loop-delta successor that preserves the
exact K=1 base path. Mechanics/data failures and infeasible gates require repair/review; a readable but
unused state routes to the controlled interface successor, and a sample-more-only loss triggers neither
because LoRA has then already formed the representation.

## 2026-07-12 — Integrated Audit Closure

- Added a dedicated pilot-validation seed/split and removed the last pre-promotion read of confirmatory validation rows.
- Pinned an exact confirmatory-config digest; every model-bearing entry rejects smoke/reduced geometry, and nonconfirmatory analysis cannot emit evidence.
- Removed dead mixed-interface scalars and the orphaned static-LoRA arm.
- Made pilot promotion require complete K=4, joint-holdout, and bidirectional-swap diagnostics without requiring favorable diagnostic signs.
- Implemented every documented verdict label, full receipt identities, current lock/source checks, exact immutable row pairing, and correct pair-clustered swap inference.
- Reanalyze raw sample-more allocations, parses, labels, totals, and by-depth natural-close/parse/cap rates.
- Final validation: 42 tests pass, Python compilation passes, and `git diff --check` is clean.

No fresh CPU smoke/data generation, model load, GPU call, training, evaluation, or benchmark access occurred.

## 2026-07-12 — First Live G0 Attempt

- Rebuilt the exact pinned Transformers environment and compiled `causal-conv1d==1.6.2.post1`.
- Fresh CPU smoke, all 41 tests, and the complete source-bound corpus passed before model loading.
- The first live G0 loaded only `Qwen/Qwen3.5-4B` at the pinned revision, then stopped before issuing
  a receipt or starting training: the smoke harness reused a K=4-encoded target tensor for its K=1
  Carry/Bag equality forward, producing a 1-versus-4 state-loss shape error.
- Fixed the harness to encode and use a dedicated K=1 batch for both equality and direct-model parity;
  added a static regression assertion. Because runtime source is identity-bound, the corpus and G0
  receipt must be regenerated/reissued before proceeding.
- Reissued the full source-bound corpus and reran G0. `MODEL_SMOKE_PASS`: K=1 direct parity `0.0`,
  Carry/Bag K=1 difference `0.0`, identical 16,800,796-parameter/value receipts, nonzero finite LoRA,
  state, step, and sufficiency gradients in both arms, finite worst-format K=12, and 11.21 GiB peak
  allocation. No scientific claim is licensed; seed-7401 pilot is next.
- Completed both fixed 300-step seed-7401 pilot trainings and their dedicated evaluations. Training
  receipts matched exactly. Before a promotion verdict was written, analysis failed closed because
  `_deployment_comparison` applied the full seed set (7411–7413) to pilot seed 7401. This is an
  analysis phase-dispatch bug, so the otherwise complete attempt is preserved but invalidated; its
  chance-like state metrics do not license either a LoRA conclusion or the capacity successor.
- Fixed pilot analysis to prohibit and skip both deployment/sample-more comparators, and added a
  synthetic end-to-end pilot regression that would fail if full deployment logic is entered. The
  source-bound contract requires fresh data, G0, training, and evaluation rather than receipt edits.
- Reissued the source-bound CPU/data receipts and passed the fixed-source G0 retry. The canonical
  `MODEL_SMOKE_PASS` records exact direct and Carry/Bag K=1 parity, identical 16,800,796-parameter
  receipts, nonzero finite gradients in every registered trainable component, finite K=12 forwards,
  and 11.02 GiB peak allocation. This is a mechanics gate only; the fresh seed-7401 pilot pair is next.

## 2026-07-13 — Fixed-Source LoRA Pilot Verdict

- Completed fresh canonical Carry and Bag pilot trainings at the sole registered pilot seed 7401,
  each for the fixed 300 steps. Initialization and training receipts matched exactly: ordered-row
  digest `97813bf9a2c7b81cf55db1a405e8e999e7e4bf953b2d50434a007140019b0e4f`,
  2,594,937 prompt tokens, and 145,316,472 decoder-layer-token applications per arm.
- Evaluated the fixed final checkpoints on all 256 pilot depth tasks at K=4 and matched depth, all
  256 pilot joint-holdout tasks, and all 64 counterfactual pairs in both directions. Both checkpoints
  retained exact direct-model K=1 parity (`0.0`). All source, config, lock, data, phase, step, seed,
  checkpoint, raw-row, and swap hashes passed reanalysis.
- The analyzer emitted terminal LoRA verdict `PILOT_MECHANISM_MISS`. Seven of eight promotion checks
  passed; the only failure was joint-state sufficiency. Carry joint node+phase+checksum step accuracy
  was `0.0045947759645059705` against the frozen `0.40` threshold, while node step accuracy was
  `0.06419115958851762`.
- Answer-level Carry minus Bag was `+0.04296875` with pilot 95% interval
  `[-0.0078125, 0.09375]`; unseen-K gain over K=4 was `+0.01171875`
  `[-0.03515625, 0.05859375]`. The joint holdout was positive (`+0.05078125`,
  `[0.0078125, 0.09765625]`), but swaps were noncausal: donor-follow gain `+0.0078125`
  `[-0.0234375, 0.0390625]` and donor-follow minus recipient-preserve `-0.0546875`.
- Stopped this experiment at G1 as preregistered. No confirmation seeds, edge-cut confirmation, text
  baseline, or sample-more calls were run. The earlier analysis-dispatch attempt remains preserved and
  invalidated; it is not pooled with the canonical result.
- This is a valid deep-state-formation failure rather than a mechanics/data/infeasible-gate stop.
  Preregistration section 10 therefore mandates creating and executing a fresh successor with
  zero-initialized full-rank deltas active only on extra R applications in layers 12–19, preserving
  the frozen base first pass/coda and exact K=1 path.

## 2026-07-13 — mandatory capacity successor raw result

- The fresh `qwen35_4b_state_carry_vs_state_bag_fullrank_delta` experiment completed its exact-row,
  held-fixed seed-7401 pilot with 892,272,640 direct full-rank delta parameters.
- Its analyzer emitted `PILOT_STATE_FORMATION_MISS`: joint state accuracy 0.00277 versus the 0.40 gate,
  Carry minus Bag -0.0156, negative unseen-K scaling, and noncausal swaps.
- The raw artifacts and label are preserved. The post-result science audit below retracts the initial
  interpretation that this single pilot closed LoRA rank.

## 2026-07-13 — post-result capacity audit correction

- The full-rank pilot failed three promotion checks simultaneously: joint-state sufficiency, positive
  Carry minus Bag, and positive effects in both query strata. Carry minus Bag was `-0.015625`; node
  was `0.0` and checksum was `-0.03125`.
- Under the successor's frozen verdict ladder, any complete pilot that fails a non-capacity promotion
  requirement has disposition `PILOT_PROMOTION_BLOCKED`. `PILOT_STATE_FORMATION_MISS` isolates the
  capacity branch only when joint-state sufficiency is the specific failure. The raw analyzer label
  therefore overstates what this mixed failure can conclude.
- Retracted the claim that the full-rank run closed LoRA capacity. Its corrected scientific
  disposition is `PILOT_PROMOTION_BLOCKED`; the metrics, summaries, and checkpoint references remain
  valid raw evidence.
- A fresh RNG-matched three-seed state-formation adjudication is mandatory, pairing rank-32 LoRA and
  full-rank extra-R deltas while holding the state-formation contract fixed. This is required to
  separate adaptation capacity from seed variation and simultaneous answer/query-gate failures.
- Neither completed pilot is licensed to advance to confirmation, edge-cut, or sample-more, and the
  capacity question must not be marked closed before the paired multi-seed adjudication.
