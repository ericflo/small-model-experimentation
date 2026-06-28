# Qwen Structured Bridge Experiment

This experiment tests a frozen Qwen encoder attached to a structured latent
executor. The trainable bridge compiles text into modular program symbols, then
an invisible executor runs the compiled program to produce the answer.

## Contents

- `src/qwen_structured_bridge_experiment.py`: task generator, Qwen feature
  extraction, bridge training, executor evaluation, and checkpointing.
- `src/analyze_qwen_structured_bridge.py`: analysis tables and figures.
- `reports/qwen_structured_bridge_experiment_log.md`: chronological experiment
  log.
- `reports/qwen_structured_bridge_paper.md`: standalone written report.
- `reports/qwen_structured_bridge_paper.html`: standalone HTML report.
- `runs/`: lightweight JSON and CSV run outputs.
- `analysis/`: generated summaries and figures.
- `checkpoint_manifest.csv`: saved checkpoint paths and sizes.

## Large Files

Trainable bridge checkpoints are stored outside this directory under:

```text
../../large_artifacts/qwen_structured_bridge/checkpoints/
```

Download this experiment directory for the research bundle. Download the large
artifact directory only when saved model weights are needed.
