# Rank-Capacity Vehicle Cell Experiment Log

## 2026-07-15 — Model-free design freeze

- Opened as the vehicle study's first single-variable cell after the
  intrinsic-tax verdict. Rank capacity is the sharpest candidate mechanism
  (independently implicated by the interference arc).
- One trainer delta (`--model-path`) lets a FRESH rank-64 adapter train on the
  frozen clean-parent composite; the corpus, exposure geometry, and gate
  design are inherited unchanged; the published rank-32 arm is re-measured on
  the same screen; the ordered three-way verdict is frozen with successor
  selection.
- Seeds `55124/58/88021`; no aggregate seed exists.
- No model, GPU, training, local, or benchmark event has run.

## 2026-07-15 — Authenticated rank-64 training

- `train-candidate` ran only after the freeze checkpoint matched `origin/main`
  with both workflows green; the fresh rank-64/alpha-128 adapter trained on
  the pinned clean-parent composite (full-weights preflight) with 1,520/1,520
  rows, 0 skipped, 190 updates; receipt/log published and pinned fail-closed.
