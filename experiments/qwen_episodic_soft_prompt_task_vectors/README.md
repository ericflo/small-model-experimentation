# Qwen Episodic Soft-Prompt Task Vectors

Standalone experiment testing whether a short continuous prefix optimized per task can improve Qwen's held-out text-transformation consistency.

## Main Run

- Run directory: `runs/main_qwen_soft_prompt_40_s6_lr001`
- Markdown report: `reports/qwen_episodic_soft_prompt_task_vectors_report.md`
- HTML report: `reports/qwen_episodic_soft_prompt_task_vectors_report.html`
- Persistent log: `experiment_log.md`

## Key CSVs

- `analysis/summary.csv`: method-level metrics.
- `analysis/method_deltas.csv`: wins/losses versus direct row-by-row inference.
- `analysis/task_details.csv`: task-level outputs and exactness.
- `analysis/row_details.csv`: row-level outputs and exactness.
- `analysis/train_log.csv`: leave-one-out soft-prompt training loss.

## Key Charts

- `analysis/figures/method_full_task_exact.png`
- `analysis/figures/row_vs_full_task.png`
- `analysis/figures/wins_losses_vs_direct.png`
- `analysis/figures/family_heatmap.png`
- `analysis/figures/train_loss_curves.png`

## Data Placement

Large benchmark data is kept outside the experiment directory under:

`/workspace/large_artifacts/qwen_episodic_soft_prompt_task_vectors`

