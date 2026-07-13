# Qwen Semantic Prefix Value Model

**Status:** finished

Standalone experiment for semantic reachability supervision over executable bytecode prefixes.

The experiment trains a frozen-Qwen bytecode compiler head, then collects partial-program actions from typed beam search. Each action receives two offline labels:

- `exact`: the action keeps the prefix equal to the canonical target program.
- `semantic`: bounded executable completion from the post-action VM state can still reach the target answer.

Separate value models are trained from these labels and used to guide typed beam search without answer access at decode time.

## Layout

```text
src/        training, evaluation, and analysis code
runs/       per-run JSON/CSV logs
analysis/   aggregate tables and figures
reports/    standalone Markdown and HTML report
```

Large checkpoints are stored outside this directory:

```text
large_artifacts/qwen_semantic_prefix_value_model/checkpoints/
```

## Reading Order

1. `reports/qwen_semantic_prefix_value_model_paper.md`
2. `analysis/summary.md`
3. `analysis/final_metrics.csv`
4. `experiment_log.md`
