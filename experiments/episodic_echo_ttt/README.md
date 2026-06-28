# Episodic ECHO-TTT

This standalone experiment tests whether temporary per-episode gradient updates on environment-observation prediction improve later decisions in a small local language model.

Small experiment files live here. Large checkpoints and caches live under `/workspace/large_artifacts/episodic_echo_ttt`.

## Main Artifacts

- `src/episodic_echo_ttt.py`: experiment runner and report generator.
- `experiment_log.md`: chronological implementation and run log.
- `runs/`: metrics and metadata for each run.
- `analysis/`: aggregate CSVs and figures.
- `reports/`: Markdown and HTML reports.
- `checkpoint_manifest.csv`: pointers to large artifacts outside this directory.
