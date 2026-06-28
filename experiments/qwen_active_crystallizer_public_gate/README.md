# Qwen Active Crystallizer Public Gate

Standalone experiment testing whether frozen Qwen probe labels can turn sparse example-level transformation behavior into a deterministic program selected from a broad candidate DSL.

## Question

Can a model's direct few-shot transformation behavior be crystallized into a single executable program that generalizes across held-out rows?

## Method

- Use public Microsoft PROSE `Transformation.Text` tasks.
- Split each task into train and held-out rows.
- Generate a broad deterministic candidate DSL from train inputs only.
- Generate synthetic probe inputs from train inputs only.
- Ask frozen `Qwen/Qwen3-4B` to label selected probes.
- Select among train-fitting candidate programs using probe-label agreement.
- Compare against examples-only selection, shuffled probe labels, and oracle candidate coverage.

## Artifacts

- Source: `src/qwen_active_crystallizer_public_gate.py`
- Reports: `reports/`
- Metrics and figures: `analysis/`
- Public benchmark checkout: `/workspace/large_artifacts/qwen_active_crystallizer_public_gate/prose-benchmarks`
