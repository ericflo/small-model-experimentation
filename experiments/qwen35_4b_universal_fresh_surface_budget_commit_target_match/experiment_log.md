# Fresh-Surface Budget-Commit Universal Curriculum Experiment Log

## 2026-07-14 — Model-free design freeze

- Opened after the residual successful-sibling terminal stop published green; this
  trial claims the program's queued bounded-computation plus canonical-answer
  commitment successor slot with a designed, non-harvested mechanism.
- Wrote `scripts/gen_fresh_curriculum.py`: the thirteen predecessor lesson
  constructors over six fresh surface pools (greek, elements, animals, ordinals,
  gems, digraphs), fresh separators/attributes/capabilities, plus the new `u_budget`
  bounded-check lesson with a planted decoy immediately past the allowance cutoff.
- Froze the corpora at construction seed 77,116: arm D `sft_fresh_designed160.jsonl`
  (160 rows, designed160 quotas, sha `e599f156...84d5`) and arm B
  `sft_fresh_budget160.jsonl` (120-row deterministic subset of arm D plus 40 budget
  lessons, 27 hits / 13 exhausts, sha `ecece8e2...9800`). Banned-vocabulary audit
  passes; `--check` regenerates byte-identically.
- Reserved fresh construction/slot-match/training/local/aggregate seeds
  `77116/55117/51/88013/78143`; aggregate seed sealed.
- Preregistered the full contract in `reports/preregistration.md`: three-axis exact
  exposure match, one training event per arm from the authenticated
  `replay_after_close` parent, a 104-task original-surface local gate with strict
  wins over parent and replay, single-winner promotion, and a four-model conditional
  aggregate pilot (strictly lift aggregate and every public family versus base;
  strictly beat replay and parent on aggregate).
- No model, GPU, training, local, or benchmark event has run.

## 2026-07-14 — Authenticated control training

- `train-control` ran only after design-freeze commit `1d82b6c7` matched
  `origin/main` with both workflows green and a clean worktree.
- `replay_repeat` trained 1,520/1,520 rows with 0 skipped over 190 updates
  (train loss 0.4063, 1,341.9 wrapper seconds); receipt and log published under
  `runs/training/` and their hashes pinned fail-closed in `train_trial.py`.
- No evaluation ran; the treatment arms remain untrained until this checkpoint
  publishes green.

## 2026-07-14 — Authenticated designed-fresh training

- `train-designed` ran only after control checkpoint `cd1cba9e` matched
  `origin/main` with both workflows green and a clean worktree.
- `designed_fresh` trained 1,520/1,520 rows with 0 skipped over 190 updates
  (train loss 0.4634, 1,334.7 wrapper seconds); receipt/log published and pinned.
- The budget arm remains untrained until this checkpoint publishes green.

## 2026-07-15 — Authenticated budget-commit training

- `train-budget` ran only after designed checkpoint `c54a5378` matched
  `origin/main` with both workflows green and a clean worktree.
- `budget_commit` trained 1,520/1,520 rows with 0 skipped over 190 updates
  (train loss 0.5106, 1,393.4 wrapper seconds); receipt/log published and pinned.
- All three arms are trained; merges are the only next stage.

## 2026-07-15 — Merge-gate pin amendment (model-free)

- The merge stage refused to open: `merge_trained_arm.py` demands its own hash
  under the receipt's `code_sha256.merge`, but the receipt generator had listed
  it as pin-deferred — an implementation inconsistency between two harness
  files, caught by the gate itself failing closed. No composite was produced.
- `merge_trained_arm.py` carries no orchestrator-filled constants, so it is now
  pinned at receipt level; the receipt regenerated with the frozen local tasks
  and oracle-free input byte-identical (`be817bd0...`, `7cba75dc...`). Bars,
  seeds, and every other frozen field are unchanged.
- No model, GPU, or evaluation event ran during the amendment.

## 2026-07-15 — Authenticated explicit composites

- `merge-arms` ran only after the amendment checkpoint `6a6f7ee7` matched
  `origin/main` with both workflows green; the PASS_CONTROL_MERGE verdict and
  the merge self-pin were both required and verified.
- All three arms merged through the pinned external merger (scale 2.0, 128/128
  nonzero modules, fingerprint-verified): tree hashes `f2aa4a76...2523`
  (replay_repeat), `93433aa2...0255` (designed_fresh), `8faf6f68...ff18`
  (budget_commit); receipts and logs under `runs/merges/`.
- The three merged-tree pins are now filled fail-closed in the local evaluator.
  The one frozen local gate event is the only next stage.
