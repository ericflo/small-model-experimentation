# Qwen Active Crystallizer Public Gate Log

## Setup

- Created fresh experiment directory: `/workspace/experiments/qwen_active_crystallizer_public_gate`
- Large artifacts directory: `/workspace/large_artifacts/qwen_active_crystallizer_public_gate`
- Public benchmark checkout is available under the large artifacts directory.
- Core question: whether Qwen probe labels can select an executable transformation program from a candidate DSL under held-out validation.
- Report format: standalone Markdown and HTML with plots.

## Smoke Debug Note

- First smoke attempt exposed a duplicate-probe loop in synthetic probe generation before producing metrics. Added an attempt cap and deterministic task seed before rerunning.

## Smoke Debug Note 2

- Second smoke attempt showed finite-map enumeration was running over concat-expanded candidates. Restricted maps to base/wrapped expressions and lowered concat breadth before rerunning.

## Run `smoke_v1`

- Started: 2026-06-27 04:36:01 UTC
- Static tasks: `35`; Qwen-probe tasks: `12`
- Completed in 23.8s.
- Candidate-oracle full-heldout coverage: 31.4%.
- Examples-only full-heldout score: 31.4%.
- Qwen-probe selected-program score: 33.3%.
- Shuffled-label selected-program score: 33.3%.
- Direct Qwen first-heldout-row score: 66.7%.

## Smoke Debug Note 3

- Third inspection showed train-signature deduplication erased candidate ambiguity. Removed that deduplication so probe labels can distinguish train-equivalent programs, and made Qwen tasks sampled by seed.

## Run `smoke_v1`

- Started: 2026-06-27 04:37:26 UTC
- Static tasks: `35`; Qwen-probe tasks: `12`
- Completed in 46.8s.
- Candidate-oracle full-heldout coverage: 45.7%.
- Examples-only full-heldout score: 31.4%.
- Qwen-probe selected-program score: 33.3%.
- Shuffled-label selected-program score: 25.0%.
- Direct Qwen first-heldout-row score: 75.0%.

## Run `pilot_v1`

- Started: 2026-06-27 04:38:41 UTC
- Static tasks: `100`; Qwen-probe tasks: `50`
- Completed in 170.9s.
- Candidate-oracle full-heldout coverage: 18.0%.
- Examples-only full-heldout score: 13.0%.
- Qwen-probe selected-program score: 18.0%.
- Shuffled-label selected-program score: 18.0%.
- Direct Qwen first-heldout-row score: 70.0%.

## Run `main_v1`

- Started: 2026-06-27 04:42:01 UTC
- Static tasks: `309`; Qwen-probe tasks: `120`
- Completed in 384.9s.
- Candidate-oracle full-heldout coverage: 29.4%.
- Examples-only full-heldout score: 22.7%.
- Qwen-probe selected-program score: 25.0%.
- Shuffled-label selected-program score: 22.5%.
- Direct Qwen first-heldout-row score: 70.0%.

## Strict Direct-Qwen Full-Heldout Diagnostic

- Model: `Qwen/Qwen3-4B`
- Tasks: `40`; held-out rows: `458`
- Row exact: 79.9% (366/458).
- Full-task exact: 37.5% (15/40).
- Full-task exact requires every held-out row for the task to be answered exactly.
