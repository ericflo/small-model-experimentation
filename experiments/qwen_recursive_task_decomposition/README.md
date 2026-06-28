# Qwen Recursive Task Decomposition

Standalone experiment testing whether recursive task decomposition improves task-level consistency on public text-transformation tasks.

## Question

Given a few input-output examples, can a solver decompose the transformation into reusable substeps and then apply the same decomposition consistently across all held-out rows?

## Main Arms

- `direct_qwen`: frozen Qwen answers each held-out row directly from the examples.
- `locked_rule_qwen`: frozen Qwen first writes a reusable rule/decomposition, then answers every held-out row while conditioned on that same rule.
- `shuffled_rule_qwen`: same as `locked_rule_qwen`, but uses a rule from another task.
- `static_mono_examples`: shortest monolithic train-fitting expression from a deterministic expression library.
- `static_mono_oracle`: held-out oracle over monolithic train-fitting expressions.
- `static_recursive_examples`: shortest recursive output-template decomposition that fits train examples.
- `static_recursive_oracle`: held-out oracle over recursive train-fitting decompositions.
- `static_recursive_shuffled`: recursive decomposition fit on rotated train labels, then evaluated on the real held-out labels.

## Layout

- `src/qwen_recursive_task_decomposition.py`: experiment runner, analysis, and report generator.
- `runs/`: raw per-run outputs.
- `analysis/`: consolidated CSVs and figures.
- `reports/`: Markdown and HTML reports.

