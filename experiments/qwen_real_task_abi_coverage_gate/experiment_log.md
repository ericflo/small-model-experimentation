# Qwen Real Task ABI Coverage Gate Log

## Setup

- Created fresh experiment directory: `/workspace/experiments/qwen_real_task_abi_coverage_gate`
- Large artifacts directory: `/workspace/large_artifacts/qwen_real_task_abi_coverage_gate`
- Core question: whether a frozen office ABI covers real-style deterministic tasks that are not generated from that ABI.
- Report format: standalone Markdown and HTML with plots.

## Iteration Notes

- Initial smoke exposed a leakage risk in date tasks: an ISO-date task could be solved by copying an unrelated interval `end` field. Fixed by separating single-date rows from interval rows before pilot/main.
- Initial pilot attempted depth-3 brute-force expressions and was interrupted after it proved too slow for a coverage gate. Fixed by bounding the gate to depth-1/2 programs, which covers direct primitives and short office compositions without turning the experiment into a synthesizer benchmark.

## Run `smoke_v1`

- Started: 2026-06-27 03:15:25 UTC
- Variants: `core,office_table`
- Depths: `1,2`
- Tasks: `12`
- Completed in 19.3s.
- Primary coverage: 83.3% (10/12 tasks).
- Train-match-only: 8.3%; no-train-match: 8.3%.

## Run `pilot_v1`

- Started: 2026-06-27 03:21:32 UTC
- Variants: `core,office,office_table`
- Depths: `1,2`
- Tasks: `30`
- Completed in 25.8s.
- Primary coverage: 86.7% (26/30 tasks).
- Train-match-only: 6.7%; no-train-match: 6.7%.

## Run `main_v1`

- Started: 2026-06-27 03:23:19 UTC
- Variants: `core,office,office_table`
- Depths: `1,2`
- Tasks: `38`
- Completed in 27.0s.
- Primary coverage: 84.2% (32/38 tasks).
- Train-match-only: 5.3%; no-train-match: 10.5%.

## Final Read

- The frozen `office_table` ABI covered 84.2% of the full hand-curated task catalog at depth 2.
- Non-calibration coverage was 78.6% (22/28), held-out composition coverage was 100.0% (8/8), and held-out new-family coverage was 100.0% (6/6).
- Table coverage was 83.3% (5/6): count, sum, filtered count, and filtered sum were covered; latest-paid-row selection was not.
- Every deliberately out-of-ABI stress task failed (0/5): fiscal-year logic, discounted total, middle SKU segment, phrase slugification, and latest paid amount.
- Read: this is a guarded positive for a fixed office ABI on common deterministic transformations, not proof of open-ended coverage. The next step should use a less hand-curated corpus or freeze the ABI before evaluating a public/production-like task set.
