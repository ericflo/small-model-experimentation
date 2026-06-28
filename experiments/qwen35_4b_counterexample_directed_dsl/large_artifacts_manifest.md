# Large Artifacts Manifest

Large files are intentionally stored outside this experiment directory so the compact directory can be downloaded without checkpoints or adapter weights.

## Compact Experiment Directory

- Path: `/workspace/experiments/qwen35_4b_counterexample_directed_dsl`
- Size at audit time: about 2.8 MB.
- Contains source, configs, generated datasets, run logs, evaluation JSON, and the final report.

## Large Artifact Directory

- Path: `/workspace/large_artifacts/qwen35_4b_counterexample_directed_dsl`
- Size at audit time: about 1.4 GB.

## Model Artifacts

| Artifact | Purpose | Size |
| --- | --- | ---: |
| `/workspace/large_artifacts/qwen35_4b_counterexample_directed_dsl/models/_smoke_random_lora` | Smoke-test QLoRA adapter and tokenizer files | 445 MB |
| `/workspace/large_artifacts/qwen35_4b_counterexample_directed_dsl/models/random_trace_lora` | Random-trace trained QLoRA adapter and tokenizer files | 445 MB |
| `/workspace/large_artifacts/qwen35_4b_counterexample_directed_dsl/models/counterexample_trace_lora` | Counterexample-trace trained QLoRA adapter and tokenizer files | 445 MB |

Each adapter directory contains `adapter_model.safetensors`, adapter config, tokenizer files, training args, and experiment metadata.
