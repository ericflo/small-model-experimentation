# Sparse Support Memory Executor Experiment

**Status:** finished

This experiment tests whether exact modular belief-state execution is recovered
when the recurrent state is an explicit sparse set of support slots rather than
a fixed-width dense vector.

## Contents

- `src/sparse_support_memory_experiment.py`: task generator, sparse support executor, evaluation harness.
- `src/analyze_sparse_support_memory.py`: analysis and figure generation.
- `reports/sparse_support_memory_experiment_log.md`: chronological experiment log.
- `reports/sparse_support_memory_paper.md`: standalone written report.
- `reports/sparse_support_memory_paper.html`: standalone HTML report.
- `runs/`: JSON and CSV run outputs.
- `analysis/`: generated summaries and figures.
- `checkpoint_manifest.csv`: checkpoint paths and sizes.

Large trainable artifacts, if any, are written outside the experiment directory:

```text
../../large_artifacts/sparse_support_memory_executor/checkpoints/
```

Download this experiment directory for the normal research bundle. Download
`../../large_artifacts/sparse_support_memory_executor/` only when saved model
weights are needed.
