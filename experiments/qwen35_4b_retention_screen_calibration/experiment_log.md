# Retention-Screen Calibration Experiment Log

## 2026-07-15 — Model-free design freeze

- Opened as the SCREEN_INSTABILITY verdict's funded successor: four gates'
  pooled scatter showed same-composite retention deltas moving ±3–4 points
  between fresh screens, rivaling the ±5 band.
- Four retention-only screens frozen (seeds 88,022–88,025) with overlap
  receipts; five published composites pinned; the 20-run screen-major event
  order, the pooled-SD outputs, the band formula, the protocol tiers, and the
  historical stability flags are preregistered.
- No model, GPU, or benchmark event has run; nothing trains in this study.

## 2026-07-15 — Adversarial design review: estimand corrected pre-freeze

- Three-lens review (contract, statistics, fail-closed) with adversarial
  verification confirmed one MAJOR finding and refuted nothing: the draft
  derived `recommended_band` and `adjudication_protocol` from the pooled SD
  of retention-correct LEVELS, but every band this program adjudicates is a
  same-screen DELTA versus a parent — common screen-difficulty variance
  inflates the level SD yet cancels exactly in deltas, while independent
  per-arm noise makes the delta SD ~√2 × the level SD, so the draft's
  outputs were calibrated against the wrong noise process in an unknowable
  direction.
- Corrected before any commit or model event: the governing estimand is now
  `delta_sd_pooled` (pooled across-screen sample SD of the per-screen
  delta-vs-clean_parent series over the four non-parent arms, ddof=1); the
  level SD stays in the receipt descriptively. Preregistration, code,
  config, README, and tests amended together; a regression test pins the
  cancellation case (levels wobbling ±4 with a constant −5 delta must read
  band 5 / single_screen, not band 7 / pooled_k2).
- 59/59 unit tests green after the amendment.

## 2026-07-15 — Calibration event (the only model event) and closure

- CI green on the freeze commit; `run.py --stage local` executed the 20
  authenticated engine runs in the frozen screen-major order; every
  composite tree recomputed and matched its receipt at the boundary.
- Readings: `delta_sd_pooled` 4.27 → `recommended_band` 9 and
  `adjudication_protocol` `pooled_k3`; level SD 4.81 (descriptive); all
  five historical single-screen tax readings fall inside their arms'
  pooled ± 2·SD intervals; pooled deltas −3.75 (axis160_direct), −2.25
  (hygiene_explore_direct), −0.75 (axis160_r64), −0.75 (replay_clean).
- Vehicle, descriptive: rank-64 −0.75 vs rank-32 −3.75 (+3.0, within
  noise).
- Closure: the ±5 single-screen band was ~1.2 SD wide; ±5 on a pooled
  three-screen mean is correctly sized (2 × 4.27/√3 = 4.9), which is the
  frozen protocol going forward. The 5–10-point per-dose tax reading
  revises to 1–4 points pooled.
