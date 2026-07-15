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
