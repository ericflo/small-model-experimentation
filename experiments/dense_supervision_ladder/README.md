# Dense Supervision Ladder Experiment

**Status:** finished

This experiment tests which training signal is sufficient for a recurrent model with a fixed-width dense hidden state to learn modular belief-state execution.

## Contents

- `src/dense_supervision_ladder_experiment.py`: training, probing, and evaluation harness.
- `src/analyze_dense_supervision_ladder.py`: analysis and figure generation.
- `reports/dense_supervision_ladder_experiment_log.md`: chronological experiment log.
- `reports/dense_supervision_ladder_paper.md`: standalone written report.
- `reports/dense_supervision_ladder_paper.html`: standalone HTML report.
- `runs/`: JSON and CSV run outputs.
- `analysis/`: generated summaries and figures.
- `checkpoint_manifest.csv`: checkpoint paths and sizes.

Checkpoints are written outside the experiment directory under:

```text
../../large_artifacts/dense_supervision_ladder/checkpoints/
```

Download this experiment directory for the normal research bundle. Download
`../../large_artifacts/dense_supervision_ladder/` only when saved model
weights are needed.
