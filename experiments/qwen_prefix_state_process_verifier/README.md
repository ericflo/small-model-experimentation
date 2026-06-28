# Qwen Prefix-State Process Verifier

Standalone experiment for prefix-level verification and typed beam search over executable bytecode.

The experiment asks whether a learned verifier over partial programs and VM state can close the gap between greedy bytecode decoding and answer-verified local search. Qwen is used as the prompt encoder, a bytecode compiler head proposes opcode/argument distributions, and a prefix-state verifier reranks typed beam-search prefixes.

## Layout

```text
src/        training, evaluation, and analysis code
runs/       per-run JSON/CSV logs
analysis/   aggregate tables and figures
reports/    standalone Markdown and HTML report
```

Large checkpoints are stored outside this directory:

```text
large_artifacts/qwen_prefix_state_process_verifier/checkpoints/
```

## Reading Order

1. `reports/qwen_prefix_state_process_verifier_paper.md`
2. `analysis/summary.md`
3. `analysis/final_metrics.csv`
4. `experiment_log.md`
