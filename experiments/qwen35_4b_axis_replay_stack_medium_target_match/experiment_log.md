# Axis-on-Replay Stack Experiment Log

## 2026-07-14 — Model-free design freeze

- Opened after the goal-gap axis experiment closed green (first local
  promotion; pilot negative with disjoint family flips between the axis
  candidate and its replay control). This trial claims both queued directions:
  stack the axis install on the 0.5081 replay-compounded parent, and measure
  round-two replay compounding inside the same event.
- Inherited the frozen 160-row axis corpus byte-identically
  (`e7a95d73...686e`); reserved fresh slot/training/gate/aggregate seeds
  `55119/53/88015/78145`; pilot preregistered at the MEDIUM tier, where the
  all-families goal has passed 8 of 92 historical events.
- No model, GPU, training, local, or benchmark event has run.

## 2026-07-14 — Authenticated control training

- `train-control` ran only after freeze commit `bcce7472` matched `origin/main`
  with both workflows green and a clean worktree.
- `replay_squared` trained 1,520/1,520 rows with 0 skipped over 190 updates;
  receipt/log published and pinned fail-closed. The candidate arm remains
  untrained until this checkpoint publishes green.

## 2026-07-15 — Authenticated candidate training

- `train-candidate` ran only after control checkpoint `f4fa0701` matched
  `origin/main` with both workflows green and a clean worktree.
- `axis_on_replay` trained 1,520/1,520 rows with 0 skipped over 190 updates;
  receipt/log published and pinned fail-closed. Merges are the only next stage.

## 2026-07-15 — Authenticated explicit composites

- `merge-arms` ran only after candidate checkpoint `02155e08` matched
  `origin/main` with both workflows green; PASS_CONTROL_MERGE and the merge
  self-pin were required and verified.
- Both arms merged (scale 2.0, 128/128 nonzero modules, fingerprint-verified);
  merged-tree pins filled fail-closed in the evaluator. The one frozen 144-task
  gate event at seed 88,015 is the only next stage.

## 2026-07-15 — Gate event: no promotion; experiment closed

- The frozen gate event ran from merge checkpoint `7183fa9e`: three
  authenticated engine runs over the 144-row input at seed 88,015.
- Axis holdout: candidate 24/40, parent 18, replay_squared 15. Per-kind
  candidate/parent/squared: explore 5/4/7, hygiene 9/5/5, protocol 8/8/3,
  tracefix 2/1/0. Retention: 64/98/6 vs 65/92/12 vs 64/86/18.
- Nine of ten checks passed; the 3-of-4 kind-breadth bar failed (protocol tied
  at the parent ceiling for the second consecutive experiment; explore lost to
  the control's 7/10). No promotion; seed 78,145 permanently sealed per the
  frozen contract; no benchmark event ran.
