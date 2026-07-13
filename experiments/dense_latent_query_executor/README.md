# Dense Latent Query Executor Experiment

**Status:** finished

This experiment tests whether a recurrent model with a fixed-width dense hidden state can learn modular belief-state execution from one sampled final query value per example.

## Contents

- `src/dense_latent_query_executor_experiment.py`: training, probing, and evaluation harness.
- `src/analyze_dense_latent_query_executor.py`: analysis and figure generation.
- `reports/dense_latent_query_executor_paper.md`: standalone writeup.
- `reports/dense_latent_query_executor_paper.html`: standalone HTML report.
- `reports/dense_latent_query_executor_experiment_log.md`: chronological experiment log.
- `runs/`: JSON and CSV run outputs.
- `analysis/`: generated summaries and figures.
- `checkpoint_manifest.csv`: paths and sizes for saved checkpoints.

Checkpoints are written outside the experiment directory under:

```text
../../large_artifacts/dense_latent_query_executor/checkpoints/
```

Download this experiment directory for the normal research bundle. Download
`../../large_artifacts/dense_latent_query_executor/` only when saved model
weights are needed.
