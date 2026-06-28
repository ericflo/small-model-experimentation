# Counterexample-Guided Ephemeral Program

This standalone experiment tests whether a local language model can be made more task-consistent by inducing a task-local executable program and selecting it with synthetic disagreement probes.

## Hypothesis

For public text-transformation tasks, direct row-by-row inference often has useful row-level semantic competence but can be inconsistent across held-out rows. A task-local executable program can provide consistency, but visible examples alone are too weak and can select brittle train-fitting programs. The experiment adds synthetic counterexample rows where candidate programs disagree, labels those rows with the model, and selects or routes candidate programs against the expanded label set.

## Methods

- `direct_qwen_row`: answer held-out rows independently from visible examples.
- `direct_qwen_batch`: answer all held-out rows in one JSON generation.
- `program_visible`: select the shortest generated program that passes visible examples.
- `program_ceg`: select a visible-passing program that also matches model-labeled disagreement probes.
- `program_ceg_gated`: use the selected program only when probe support is strong; otherwise fall back to direct row answers.
- `program_ceg_router`: if no single candidate explains all visible and probe labels, fit a simple two-branch router over candidate programs.
- Controls: random-probe selection and shuffled synthetic-probe labels.
- Diagnostics: hidden candidate oracle and train-pass rate.

## Metrics

The primary metric is strict full-task exact on held-out rows. A task counts only if every held-out row is exactly correct. Secondary metrics are row exact, number of train-passing candidates, selected-program probe score, synthetic-label consensus, and task-level wins/losses versus direct inference.

## Artifacts

Generated outputs live under `runs/<run_name>/`, mirrored analysis tables under `analysis/`, charts under `analysis/figures/`, and reports under `reports/`. Large benchmark files are referenced from `/workspace/large_artifacts/qwen_counterexample_guided_ephemeral_program`.
