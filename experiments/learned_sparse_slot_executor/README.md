# Learned Sparse Slot Executor Experiment

This experiment tests whether a neural recurrent runtime can learn to use an
explicit slot memory for modular belief-state execution.

## Contents

- `src/learned_sparse_slot_experiment.py`: task generator, trainable slot executor, checkpointing, and evaluation harness.
- `src/analyze_learned_sparse_slot.py`: analysis and figure generation.
- `reports/learned_sparse_slot_experiment_log.md`: chronological experiment log.
- `reports/learned_sparse_slot_paper.md`: standalone written report.
- `reports/learned_sparse_slot_paper.html`: standalone HTML report.
- `runs/`: JSON and CSV run outputs.
- `analysis/`: generated summaries and figures.
- `checkpoint_manifest.csv`: checkpoint paths and sizes.

Checkpoints are written outside the experiment directory under:

```text
../../large_artifacts/learned_sparse_slot_executor/checkpoints/
```

Download this experiment directory for the normal research bundle. Download
`../../large_artifacts/learned_sparse_slot_executor/` only when saved model
weights are needed.
