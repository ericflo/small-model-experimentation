# Experiment Log

## Setup

- Created a fresh standalone experiment directory.
- Large artifacts are kept under `/workspace/large_artifacts/qwen_oracle_distilled_acquisition_policy`.
- Primary metric: strict full-task exact on held-out rows.
- Secondary metrics: row exact, candidate utility-label quality, acquisition policy wins/losses, and oracle headroom.
- Core intervention: train a cross-validated acquisition-row scorer from counterfactual downstream utility labels, then use that scorer to pick clarifying examples on held-out tasks.


### Run `smoke_no_qwen`
- Tasks: 4
- Generation records: 104
- `base_plain`: 0.0% full-task exact.
- `learned1_plain`: 0.0% full-task exact.
- `learned3_plain`: 0.0% full-task exact.
- `order3_plain`: 0.0% full-task exact.
- `random3_plain`: 0.0% full-task exact.
- `qwen_choose3_plain`: 0.0% full-task exact.
- `learned3_shuffled_labels`: 0.0% full-task exact.
- `oracle_single`: 0.0% full-task exact.

### Run `smoke_no_qwen_v2`
- Tasks: 4
- Generation records: 104
- `base_plain`: 0.0% full-task exact.
- `learned1_plain`: 0.0% full-task exact.
- `learned3_plain`: 0.0% full-task exact.
- `order3_plain`: 0.0% full-task exact.
- `random3_plain`: 0.0% full-task exact.
- `qwen_choose3_plain`: 0.0% full-task exact.
- `learned3_shuffled_labels`: 0.0% full-task exact.
- `oracle_single`: 0.0% full-task exact.

### Run `pilot_qwen_6`
- Tasks: 6
- Generation records: 264
- `base_plain`: 66.7% full-task exact.
- `learned1_plain`: 66.7% full-task exact.
- `learned4_plain`: 50.0% full-task exact.
- `order4_plain`: 66.7% full-task exact.
- `random4_plain`: 66.7% full-task exact.
- `qwen_choose4_plain`: 66.7% full-task exact.
- `learned4_shuffled_labels`: 16.7% full-task exact.
- `oracle_single`: 66.7% full-task exact.

### Run `main_v1`
- Tasks: 30
- Generation records: 1320
- `base_plain`: 50.0% full-task exact.
- `learned1_plain`: 50.0% full-task exact.
- `learned4_plain`: 56.7% full-task exact.
- `order4_plain`: 60.0% full-task exact.
- `random4_plain`: 63.3% full-task exact.
- `qwen_choose4_plain`: 60.0% full-task exact.
- `learned4_shuffled_labels`: 26.7% full-task exact.
- `oracle_single`: 66.7% full-task exact.
