# Cyclic Transition Ladder Experiment

**Status:** finished

This experiment tests which transition inductive bias is sufficient for a
recurrent slot model to learn modular belief-state execution.

## Contents

- `src/cyclic_transition_ladder_experiment.py`: task generator, transition
  ladder models, checkpointing, and evaluation harness.
- `src/analyze_cyclic_transition_ladder.py`: analysis and figure generation.
- `reports/cyclic_transition_ladder_experiment_log.md`: chronological
  experiment log.
- `reports/cyclic_transition_ladder_paper.md`: standalone written report.
- `reports/cyclic_transition_ladder_paper.html`: standalone HTML report.
- `runs/`: JSON and CSV run outputs.
- `analysis/`: generated summaries and figures.
- `checkpoint_manifest.csv`: checkpoint paths and sizes.

Checkpoints are written outside the experiment directory under:

```text
../../large_artifacts/cyclic_transition_ladder/checkpoints/
```

Download this experiment directory for the normal research bundle. Download
`../../large_artifacts/cyclic_transition_ladder/` only when saved model weights
are needed.
