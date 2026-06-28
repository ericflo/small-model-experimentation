# Qwen Budgeted Action-Value Compiler

Standalone experiment for action-value learning over executable bytecode prefixes.

The experiment trains a frozen-Qwen bytecode compiler head, then collects partial-program actions from typed beam search. Each action receives three offline targets:

- `exact`: the action keeps the prefix equal to the canonical target program.
- `found`: bounded executable completion from the post-action VM state can still find a correct completion.
- `qvalue`: a graded return based on the rank and log-probability margin of the best correct completion under the remaining search budget.

Separate value models are trained from these targets and used to guide typed beam search without answer access at decode time.

## Layout

```text
src/        training, evaluation, and analysis code
runs/       per-run JSON/CSV logs
analysis/   aggregate tables and figures
reports/    standalone Markdown and HTML report
```

Large checkpoints are stored outside this directory:

```text
large_artifacts/qwen_budgeted_action_value_compiler/checkpoints/
```

## Reading Order

1. `reports/qwen_budgeted_action_value_compiler_paper.md`
2. `analysis/summary.md`
3. `analysis/final_metrics.csv`
4. `experiment_log.md`
