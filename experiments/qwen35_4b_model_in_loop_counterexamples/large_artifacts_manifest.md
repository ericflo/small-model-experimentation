# Large Artifact Manifest

Large model artifacts for this experiment are intentionally stored outside the compact experiment directory.

## Compact Directory

- Path: `/workspace/experiments/qwen35_4b_model_in_loop_counterexamples/`
- Size after report generation: about 6.2 MB
- Contents: configs, source, scripts, generated datasets, run logs, evaluation JSON, mining report, and final report.

## Large Artifact Directory

- Path: `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/`
- Size after training: about 1.4 GB
- Contents: LoRA adapter outputs, tokenizer snapshots, trainer metadata, and `checkpoint-60` snapshots.

## Adapter Directories

| Adapter | Path | Size |
| --- | --- | ---: |
| Seed adapter | `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/models/seed_lora` | 445 MB |
| Static bridge adapter | `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/models/static_bridge_lora` | 445 MB |
| Model-loop bridge adapter | `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/models/model_loop_lora` | 445 MB |

## Checkpoint Snapshots

| Checkpoint | Path | Size |
| --- | --- | ---: |
| Seed adapter checkpoint | `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/models/seed_lora/checkpoint-60` | 264 MB |
| Static bridge checkpoint | `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/models/static_bridge_lora/checkpoint-60` | 264 MB |
| Model-loop checkpoint | `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/models/model_loop_lora/checkpoint-60` | 264 MB |

## Largest Files

| File | Size |
| --- | ---: |
| `models/seed_lora/adapter_model.safetensors` | 169,907,160 bytes |
| `models/static_bridge_lora/adapter_model.safetensors` | 169,907,160 bytes |
| `models/model_loop_lora/adapter_model.safetensors` | 169,907,160 bytes |
| `models/seed_lora/tokenizer.json` | 19,989,424 bytes |
| `models/static_bridge_lora/tokenizer.json` | 19,989,424 bytes |
| `models/model_loop_lora/tokenizer.json` | 19,989,424 bytes |

