# Structured Slot Initializer Ladder Experiment

This experiment tests which initializer structure is sufficient to populate a
sparse modular belief support before an exact recurrent transition executes the
program.

## Contents

- `src/structured_slot_initializer_ladder_experiment.py`: task generator,
  initializer ladder models, exact transition, checkpointing, and evaluation
  harness.
- `src/analyze_structured_slot_initializer_ladder.py`: analysis and figure
  generation.
- `reports/structured_slot_initializer_ladder_experiment_log.md`:
  chronological experiment log.
- `reports/structured_slot_initializer_ladder_paper.md`: standalone written
  report.
- `reports/structured_slot_initializer_ladder_paper.html`: standalone HTML
  report.
- `runs/`: JSON and CSV run outputs.
- `analysis/`: generated summaries and figures.
- `checkpoint_manifest.csv`: checkpoint paths and sizes.

Checkpoints are written outside the experiment directory under:

```text
../../large_artifacts/structured_slot_initializer_ladder/checkpoints/
```

Download this experiment directory for the normal research bundle. Download
`../../large_artifacts/structured_slot_initializer_ladder/` only when saved
model weights are needed.
