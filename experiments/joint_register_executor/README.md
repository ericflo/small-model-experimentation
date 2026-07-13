# Joint Register Executor Experiment

**Status:** finished

This experiment tests latent recurrent execution for two-register modular programs with cross-register operations such as `A=A+B` and `B=B-A`.

## Contents

- `src/joint_register_executor_experiment.py`: training and evaluation harness.
- `src/analyze_joint_register_executor.py`: aggregates run metrics and regenerates figures.
- `reports/joint_register_executor_paper.md`: standalone paper-style report.
- `reports/joint_register_executor_paper.html`: HTML version of the report.
- `reports/joint_register_executor_experiment_log.md`: chronological experiment log.
- `runs/`: JSON and CSV run outputs.
- `analysis/`: generated summaries and figures.
- `checkpoint_manifest.csv`: saved checkpoint list and sizes.

Checkpoints are written outside the experiment directory under:

```text
../../large_artifacts/joint_register_executor/checkpoints/
```

## Intended Download Split

Download this experiment directory for code, reports, figures, and metrics. Download the matching `large_artifacts` checkpoint directory only if saved weights are needed.
