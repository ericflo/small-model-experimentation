# Latent Recurrent Executor Experiment

This controlled experiment tests whether a neural runtime can execute two-register modular programs one hidden recurrent step at a time.

## Contents

- `src/latent_executor_experiment.py`: training and evaluation script.
- `src/analyze_latent_executor.py`: regenerates analysis CSVs and figures from run metrics.
- `reports/latent_executor_paper.md`: standalone paper-style report.
- `reports/latent_executor_paper.html`: HTML version of the report.
- `reports/latent_executor_experiment_log.md`: chronological run log.
- `analysis/`: generated figures, summary Markdown, and analysis CSVs.
- `runs/`: small JSON/CSV run outputs. Checkpoint `.pt` files are not stored here.
- `checkpoint_manifest.csv`: list of saved checkpoints stored outside this directory.

## Large Files

Model checkpoints are stored at:

```text
../../large_artifacts/latent_executor/checkpoints/
```

Download that directory only if you need to load saved model weights. The paper, plots, and analysis tables do not require it.

## Useful Commands

Regenerate analysis outputs from the stored run metrics:

```bash
python experiments/latent_executor/src/analyze_latent_executor.py
```

Run a new experiment from this experiment directory or from the workspace root, passing an explicit `--output_dir` if you want a named run.

