# Idea Intake: Retention-Screen Calibration Study

## Program Fit

- Program: `agentic_breadth_installation`.
- Existing or new program: existing.
- Closest scorecard: Agentic Breadth Installation.
- Related-work discovery: `make related QUERY="retention screen seed variance calibration bands pooled adjudication"`.

## Prior Evidence

- Anchor: the rank-capacity cell's SCREEN_INSTABILITY verdict — the known −9
  re-measured at −5 — and the pooled scatter across four gates (same-composite
  retention deltas moving ±3–4 points between fresh screens), which rivals the
  ±5 band every dose/vehicle adjudication has used.
- Closest near-duplicate: none — every prior gate consumed its screen for a
  treatment verdict; none measured the screen itself.

## Novelty Claim

The line's first instrument-calibration study: five published composites
re-measured across four fresh retention-only screens (20 authenticated
eval events, zero training), yielding direct per-arm screen variance, a
data-derived band, and a frozen adjudication protocol for every future cell.

## Mechanism

The 104-task screen is a binomial-ish draw per arm; its seed-to-seed SD is
directly measurable with repeated fresh screens. Bands below the measured
noise adjudicate luck; bands derived from it adjudicate effects.

## Control Plan

- All arms are published, weight-authenticated composites; screens are
  overlap-receipted against all prior gates and each other; normalization
  unchanged; frozen screen-major run order.
- Outputs preregistered: pooled within-arm SD, recommended band
  (⌈2·SD⌉, min 5), adjudication protocol tier, and stability flags for the
  historical single-screen readings.
- No promotion, no benchmark seed, no claims.
- Hidden boundary: `benchmarks/` unread.

## Evidence Output

- Program evidence: the measured screen noise and the frozen banding protocol;
  retroactive stability flags for the intrinsic-tax chain's single-screen
  readings.
- Stop condition: single-shot; the calibration governs all future retention
  adjudications, and the paused vehicle question resumes under it.

## Decision

- Run experiment: model-free construction, then ONE eval-only stage (20
  engine runs).
- Defer: the stage until the freeze checkpoint is committed, pushed, green.

Fresh screen seeds are `88022/88023/88024/88025`; no training or aggregate
seeds exist.
