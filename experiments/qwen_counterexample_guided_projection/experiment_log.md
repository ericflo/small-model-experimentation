# Qwen Counterexample-Guided Projection Experiment Log

## Objective

Test whether a model's row-level transformation guesses can be crystallized into a stable task-level deterministic transducer by generating counterexample-style probe inputs, labeling those probes with the model, and selecting a train-consistent program using a noise-aware objective.

The experiment is standalone: it uses public text-transformation tasks, writes all local outputs under this directory, and stores any large reusable artifacts under `/workspace/large_artifacts/qwen_counterexample_guided_projection`.

## Initial Plan

1. Build a standalone runner for public text-transformation tasks.
2. Enumerate deterministic candidate expressions that exactly match training examples.
3. Generate task-local probe inputs by mutating and recombining training inputs.
4. Label probes with Qwen row inference.
5. Select among train-consistent expressions using probe-label agreement plus a complexity penalty.
6. Compare against train-only deterministic selection, shuffled probe labels, direct Qwen held-out outputs, and simple output-level ensemble/majority baselines when available.
7. Generate CSVs, charts, a Markdown report, and an HTML report.

## Run Notes

### 2026-06-27 06:48 UTC - Smoke: `smoke_no_qwen`

- Created a new standalone experiment directory.
- Ran a 6-task no-Qwen smoke test to verify benchmark loading, deterministic candidate enumeration, CSV writing, chart generation, Markdown report, and HTML report.
- Finding: the scaffold worked, but the initial candidate deduplication would have collapsed all train-consistent candidates into one because train-consistent expressions share the same train signature. Fixed that before real probe labeling.

### 2026-06-27 06:51 UTC - Candidate-class iteration

- Added token-span extraction and simple affix expressions.
- Reran a no-Qwen smoke test.
- Finding: the broader grammar helped some substring-style tasks but many public tasks still had zero train-consistent candidates.

### 2026-06-27 06:53 UTC - Matched-split diagnostic: `diagnostic_no_qwen_matched`

- Aligned the task split with the cached direct row-by-row baseline: seed `20260627`, 40 tasks, 4 train rows, up to 6 held-out rows.
- Train-only deterministic selection reached 20.0% full-task exact and 21.2% row exact.
- Direct row-by-row baseline reached 50.0% full-task exact and 72.1% row exact.
- Candidate availability was the main bottleneck: 29/40 tasks had zero train-consistent candidates; 11/40 had at least one candidate.

### 2026-06-27 06:56 UTC - Qwen probe pilot: `pilot_qwen_6`

- Queried Qwen on 24 generated probe rows across 6 tasks.
- Probe labels were often sensible, especially for date/month and numeric-format tasks.
- The projection still could not act on most tasks because there were no train-consistent deterministic candidates.
- Added a high-agreement gated projection arm that falls back to direct row inference unless the selected program strongly agrees with probe labels.
- Added a concrete-output majority baseline over existing row/batch prompt variants.

### 2026-06-27 07:01 UTC - Main run: `main_qwen_probe_40`

- Queried Qwen for 320 generated probe labels: 40 tasks x 8 probes.
- Main results:
  - `direct_row_by_row`: 72.1% row exact, 50.0% full-task exact.
  - `qwen_projection`: 21.2% row exact, 20.0% full-task exact.
  - `train_only`: 21.2% row exact, 20.0% full-task exact.
  - `qwen_projection_shuffled`: 21.2% row exact, 20.0% full-task exact.
  - `qwen_projection_gated_direct`: 71.5% row exact, 50.0% full-task exact.
  - `output_majority`: 71.5% row exact, 47.5% full-task exact.
- Interpretation:
  - Real Qwen probe labels did not separate from shuffled/random probe controls on full-task accuracy.
  - The ungated projection was limited by deterministic hypothesis coverage.
  - The gated projection safely tied direct full-task exact but did not improve it, and it slightly reduced row exact.
  - Simple output-majority across prompt variants also did not improve over direct row-by-row.

### 2026-06-27 07:05 UTC - Report hardening

- Added candidate availability diagnostics.
- Added gated-decision diagnostics.
- Regenerated the standalone Markdown and HTML reports with six charts.
