# Large Artifacts Manifest

Large files for this experiment are intentionally stored outside the compact experiment directory.

## Large Artifact Root

`/workspace/large_artifacts/qwen35_4b_executable_program_posttraining/`

Current size observed during the run: approximately `1.4G`.

## Model Artifacts

- `models/dsl_trace_lora/`
  - First trace-conditioned executable DSL adapter.
  - Includes final adapter files and `checkpoint-60/`.

- `models/dsl_trace_and_bridge_lora/`
  - Second trace-conditioned executable DSL adapter trained with additional training-only conjunction families.
  - Includes final adapter files and `checkpoint-60/`.

- `models/_smoke_trace_lora/`
  - Tiny 8-record compatibility smoke adapter.
  - Kept only as a reproducibility artifact for the initial trainer/model-class check.

## Compact Directory

The downloadable compact directory is:

`/workspace/experiments/qwen35_4b_executable_program_posttraining/`

It contains source, configs, generated datasets, run logs, JSON reports, and the final Markdown report, but not adapter/checkpoint weights.
