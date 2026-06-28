# Experiment Log

## Setup

- Created a standalone experiment directory at `/workspace/experiments/qwen35_4b_adaptive_evidence_budget_policy`.
- Created a separate large-artifact root at `/workspace/large_artifacts/qwen35_4b_adaptive_evidence_budget_policy`.
- Objective: test Qwen3.5-4B as a deployable STOP/MORE controller for an executable verifier.

## Mechanism

- The verifier chooses the next probe by target-independent expected split over the full probe pool.
- The model sees observed executions, current survivor count, current deterministic selected program, and the expected split statistics for the best next probe.
- The model outputs:
  - `A`: STOP and commit the current selected program.
  - `B`: MORE and request one additional executable probe.
- Training labels are oracle-supervised from hidden checks: STOP once the current selected program is hidden-correct, otherwise MORE until the maximum budget.

## Planned Arms

- `fixed_budget3`
- `fixed_budget6`
- `fixed_budget10`
- `threshold_100`
- `threshold_1000`
- `oracle_stop`
- `base_budget_policy`
- `sft_budget_policy`

## Success Criterion

The SFT policy should land on or above the fixed-budget accuracy/probe Pareto curve. If fixed budgets dominate it, the useful mechanism is the inference loop with more probes, not learned stop control.

## Results

- Built 240 train records, 160 eval records, 2,640 train STOP/MORE states, and 1,760 eval STOP/MORE states.
- Trained Qwen3.5-4B QLoRA STOP/MORE adapter for 220 optimizer steps.
- Overall hidden-all accuracy / average probes:
  - `fixed_budget3`: 45.0% / 3.00
  - `fixed_budget6`: 74.4% / 6.00
  - `fixed_budget10`: 92.5% / 10.00
  - `threshold_100`: 6.9% / 1.86
  - `threshold_1000`: 5.0% / 0.94
  - `oracle_stop`: 92.5% / 4.22
  - `base_budget_policy`: 5.0% / 0.04
  - `sft_budget_policy`: 92.5% / 4.81
- By template, `sft_budget_policy` reached:
  - `pair_affine_mod`: 98.8% / 2.34 probes
  - `pair_compare_gate`: 86.2% / 7.28 probes
- Interpretation: Qwen posttraining learned a useful budget controller. It matched the accuracy of always spending ten probes while using less than half the probes on average, and it landed close to the hidden-oracle stopping efficiency.
