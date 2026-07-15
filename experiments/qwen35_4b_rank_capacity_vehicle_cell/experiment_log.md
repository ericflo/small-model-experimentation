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

## 2026-07-15 — Authenticated rank-64 composite

- `merge-candidate` ran only after the training checkpoint matched
  `origin/main` with both workflows green; the rank-64 adapter merged onto the
  clean-parent composite (scale 2.0, 128/128 modules, fingerprint-verified);
  the tree pin filled fail-closed. The one frozen three-arm verdict gate at
  seed 88,021 is the only next stage.

## 2026-07-15 — Verdict: SCREEN_INSTABILITY; cell closed

- The three-arm gate ran from the merge checkpoint: retention 69 / 64 / 62
  (parent / r32 / r64). The r32 arm's known −9 re-measured at −5, tripping the
  instability guard; no capacity inference was made. Axis: r32 21, r64 19
  (install_preserved false), parent 17.
- Pooled across the last four gates, same-composite retention deltas scatter
  ±3-4 points — the screen's seed noise is comparable to the ±5 band. The
  preregistered successor is an eval-only retention-screen calibration study.
