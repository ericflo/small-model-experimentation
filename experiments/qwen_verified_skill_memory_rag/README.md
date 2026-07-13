# Qwen Verified Skill Memory RAG

**Status:** finished

Standalone experiment testing whether train-only verified transformation examples help Qwen solve public text-transformation tasks more consistently.

## Main Run

- Run directory: `runs/main_qwen_skill_memory_40_top1`
- Markdown report: `reports/qwen_verified_skill_memory_rag_report.md`
- HTML report: `reports/qwen_verified_skill_memory_rag_report.html`
- Persistent log: `experiment_log.md`

## Key CSVs

- `analysis/summary.csv`: method-level metrics.
- `analysis/method_deltas.csv`: wins/losses versus direct row-by-row inference.
- `analysis/task_details.csv`: task-level outputs and exactness.
- `analysis/row_details.csv`: row-level outputs and exactness.
- `analysis/retrieval_details.csv`: retrieved skill cards per task.
- `analysis/retrieval_summary.csv`: retrieval quality summary.

## Key Charts

- `analysis/figures/method_full_task_exact.png`
- `analysis/figures/row_vs_full_task.png`
- `analysis/figures/family_heatmap.png`
- `analysis/figures/retrieval_family_agreement.png`
- `analysis/figures/top_retrieval_scores.png`
- `analysis/figures/wins_losses_vs_direct.png`

## Data Placement

Large benchmark data is kept outside the experiment directory under:

`/workspace/large_artifacts/qwen_verified_skill_memory_rag`

