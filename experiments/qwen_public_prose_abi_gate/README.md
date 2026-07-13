# Qwen Public PROSE ABI Gate

**Status:** finished

Standalone experiment testing whether a frozen deterministic transformation ABI covers public Microsoft PROSE `Transformation.Text` tasks under within-task held-out validation.

## Question

Does the office/transformation ABI cover an independent public benchmark under held-out validation?

## Method

- Fetch public benchmark data into `/workspace/large_artifacts/qwen_public_prose_abi_gate/prose-benchmarks`.
- Freeze a generic transformation-template ABI before evaluating the benchmark.
- For each task with enough examples, split examples into train and held-out rows.
- Select a program only if it fits train examples; count it as covered only if it also matches held-out examples.
- Report train-only fits separately as semantic/coincidence failures.

## Artifacts

- Source: `src/qwen_public_prose_abi_gate.py`
- Reports: `reports/`
- Metrics and figures: `analysis/`
- Public benchmark checkout: `/workspace/large_artifacts/qwen_public_prose_abi_gate/prose-benchmarks`
