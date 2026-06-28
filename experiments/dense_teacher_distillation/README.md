# Dense Teacher Distillation Experiment

This experiment tests whether a fixed-width dense recurrent state can learn exact modular belief-state execution when the training signal is the full teacher belief distribution at every prefix step.

## Contents

- `src/dense_teacher_distillation_experiment.py`: training, probing, checkpointing, and evaluation harness.
- `src/analyze_dense_teacher_distillation.py`: analysis and figure generation.
- `reports/dense_teacher_distillation_experiment_log.md`: chronological experiment log.
- `reports/dense_teacher_distillation_paper.md`: standalone written report.
- `reports/dense_teacher_distillation_paper.html`: standalone HTML report.
- `runs/`: JSON and CSV run outputs.
- `analysis/`: generated summaries and figures.
- `checkpoint_manifest.csv`: checkpoint paths and sizes.

Checkpoints are written outside the experiment directory under:

```text
../../large_artifacts/dense_teacher_distillation/checkpoints/
```

Download this experiment directory for the normal research bundle. Download
`../../large_artifacts/dense_teacher_distillation/` only when saved model
weights are needed.
