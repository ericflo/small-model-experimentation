# Qwen Trace Bootstrap Retention Experiment

This experiment tests whether a frozen Qwen encoder attached to a structured
latent executor can keep using a learned program interface after symbol-trace
supervision is removed.

## Contents

- `src/qwen_trace_bootstrap_retention_experiment.py`: task generator, Qwen
  feature extraction, staged bridge training, executor evaluation, and
  checkpointing.
- `src/analyze_qwen_trace_bootstrap_retention.py`: analysis tables and
  figures.
- `reports/qwen_trace_bootstrap_retention_experiment_log.md`: chronological
  experiment log.
- `reports/qwen_trace_bootstrap_retention_paper.md`: standalone written
  report.
- `reports/qwen_trace_bootstrap_retention_paper.html`: standalone HTML report.
- `runs/`: lightweight JSON and CSV run outputs.
- `analysis/`: generated summaries and figures.
- `checkpoint_manifest.csv`: saved checkpoint paths and sizes.

## Large Files

Trainable bridge checkpoints are stored outside this directory under:

```text
../../large_artifacts/qwen_trace_bootstrap_retention/checkpoints/
```

Download this experiment directory for the research bundle. Download the large
artifact directory only when saved model weights are needed.
