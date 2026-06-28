# Qwen Public PROSE ABI Gate Log

## Setup

- Created fresh experiment directory: `/workspace/experiments/qwen_public_prose_abi_gate`
- Large artifacts directory: `/workspace/large_artifacts/qwen_public_prose_abi_gate`
- Downloaded Microsoft PROSE public benchmark suite into the large artifacts directory.
- Core question: whether a frozen transformation ABI covers a less-curated public benchmark under held-out validation.
- Report format: standalone Markdown and HTML with plots.

## Run `smoke_v1`

- Started: 2026-06-27 03:43:06 UTC
- Tasks: `40`
- Train examples per task: `3`; held-out cap: `12`
- Completed in 45.4s.
- Primary coverage: 20.0% (8/40 tasks).
- Train-only: 7.5%; no-train-match: 72.5%.

## Run `pilot_v1`

- Started: 2026-06-27 03:44:21 UTC
- Tasks: `120`
- Train examples per task: `4`; held-out cap: `20`
- Completed in 179.9s.
- Primary coverage: 8.3% (10/120 tasks).
- Train-only: 5.0%; no-train-match: 86.7%.

## Run `main_v1`

- Started: 2026-06-27 03:47:49 UTC
- Tasks: `309`
- Train examples per task: `4`; held-out cap: `50`
- Completed in 511.5s.
- Primary coverage: 19.1% (59/309 tasks).
- Train-only: 3.6%; no-train-match: 77.3%.

## Frozen Qwen Direct-Answer Sample

- Model: `Qwen/Qwen3-4B`
- Sample: `60` public tasks, seed `20260627`; first `4` examples shown, one held-out query scored.
- Exact match: 73.3% (44/60).
- ABI-covered sample exact: 93.8% (15/16); ABI-missed sample exact: 65.9% (29/44).
- This baseline scores one held-out query per task, so it is diagnostic rather than directly comparable to full-task ABI coverage.
