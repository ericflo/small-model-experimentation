# Qwen Batched Transduction Consistency

**Status:** finished

Standalone experiment testing whether batched transduction improves task-level consistency on public text-transformation tasks.

## Question

Given a few input-output examples and multiple query rows, does answering all queries in one shared generation context improve full-task consistency compared with answering each row independently?

## Main Arms

- `row_by_row`: one prompt per held-out row.
- `batch_2`: held-out rows answered in batches of two.
- `batch_4`: held-out rows answered in batches of four.
- `batch_all`: all held-out rows for the task answered in one JSON list.
- `batch_all_shuffled`: all held-out rows answered in one JSON list, but query order is deterministically shuffled and then unshuffled for scoring.
- `batch_all_rule_hint`: all held-out rows answered together with an instruction to infer one rule internally before emitting the JSON list.

## Layout

- `src/qwen_batched_transduction_consistency.py`: runner and report generator.
- `runs/`: raw per-run outputs.
- `analysis/`: consolidated CSVs and figures.
- `reports/`: Markdown and HTML reports.

