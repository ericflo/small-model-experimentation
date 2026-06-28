# Experiment Log

## Setup

- Created a fresh standalone experiment directory.
- Large artifacts are kept under `/workspace/large_artifacts/qwen_learned_active_interrogation`.
- Primary metric: strict full-task exact on held-out rows.
- Secondary metrics: row exact, chosen acquisition rows, shuffled-label sensitivity, and oracle headroom over tested acquisition policies.
- Core intervention: Qwen chooses which unlabeled acquisition-pool rows should be labeled before held-out inference.


### Run `smoke_no_qwen`
- Tasks: 3
- Generation records: 180
- `base_plain`: 0.0% full-task exact.
- `qwen_choose1_plain`: 0.0% full-task exact.
- `qwen_choose2_plain`: 0.0% full-task exact.
- `random2_plain`: 0.0% full-task exact.
- `entropy2_plain`: 0.0% full-task exact.
- `qwen_choose2_shuffled_labels`: 0.0% full-task exact.

### Run `smoke_qwen_5`
- Tasks: 5
- Generation records: 300
- `base_plain`: 100.0% full-task exact.
- `qwen_choose1_plain`: 100.0% full-task exact.
- `qwen_choose2_plain`: 100.0% full-task exact.
- `random2_plain`: 80.0% full-task exact.
- `entropy2_plain`: 100.0% full-task exact.
- `qwen_choose2_shuffled_labels`: 40.0% full-task exact.

### Run `main_v1`
- Tasks: 30
- Generation records: 2550
- `base_plain`: 63.3% full-task exact.
- `qwen_choose1_plain`: 63.3% full-task exact.
- `qwen_choose4_plain`: 70.0% full-task exact.
- `random4_plain`: 66.7% full-task exact.
- `entropy4_plain`: 60.0% full-task exact.
- `qwen_choose4_shuffled_labels`: 46.7% full-task exact.
- Interpretation: revealing four extra labels improved over the two-example baseline, and corrupted labels were clearly harmful. The Qwen selector did not beat the strongest same-budget control: `order4_plain` also reached 70.0%. The diagnostic oracle over tested acquisition policies reached 80.0%, so there is still selection headroom.
