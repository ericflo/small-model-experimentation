# Query Filter Executor Experiment

**Status:** finished

This experiment tests latent recurrent execution over correlated belief states when training supervision is limited to final query answers.

## Contents

- `src/query_filter_executor_experiment.py`: training and evaluation harness.
- `src/analyze_query_filter_executor.py`: analysis and figure generation.
- `reports/query_filter_executor_paper.md`: standalone writeup.
- `reports/query_filter_executor_paper.html`: standalone HTML report.
- `reports/query_filter_executor_experiment_log.md`: chronological experiment log.
- `runs/`: JSON and CSV run outputs.
- `analysis/`: generated summaries and figures.
- `checkpoint_manifest.csv`: paths and sizes for saved checkpoints.

Checkpoints are written outside the experiment directory under:

```text
../../large_artifacts/query_filter_executor/checkpoints/
```

Download this experiment directory for the normal research bundle. Download
`../../large_artifacts/query_filter_executor/` only when saved model weights
are needed.
