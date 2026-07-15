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
