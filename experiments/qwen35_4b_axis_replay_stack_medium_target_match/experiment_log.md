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
