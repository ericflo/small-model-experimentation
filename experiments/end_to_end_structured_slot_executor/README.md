# End-to-End Structured Slot Executor Experiment

This experiment tests whether a structured slot executor can learn both initial
support formation and recurrent modular belief-state transitions in one model.

## Contents

- `src/end_to_end_structured_slot_experiment.py`: task generator, structured
  initializers, learned transition routers, checkpointing, and evaluation
  harness.
- `src/analyze_end_to_end_structured_slot.py`: analysis and figure generation.
- `reports/end_to_end_structured_slot_experiment_log.md`: chronological
  experiment log.
- `reports/end_to_end_structured_slot_paper.md`: standalone written report.
- `reports/end_to_end_structured_slot_paper.html`: standalone HTML report.
- `runs/`: JSON and CSV run outputs.
- `analysis/`: generated summaries and figures.
- `checkpoint_manifest.csv`: checkpoint paths and sizes.

Checkpoints are written outside the experiment directory under:

```text
../../large_artifacts/end_to_end_structured_slot_executor/checkpoints/
```

Download this experiment directory for the normal research bundle. Download
`../../large_artifacts/end_to_end_structured_slot_executor/` only when saved
model weights are needed.
