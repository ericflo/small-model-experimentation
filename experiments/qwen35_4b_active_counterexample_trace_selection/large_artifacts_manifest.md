# Large Artifacts Manifest

Large files are intentionally stored outside the experiment directory so this package can be downloaded without model checkpoints.

| Artifact | Location | Notes |
| --- | --- | --- |
| Sketch LoRA adapter | `/workspace/large_artifacts/qwen35_4b_active_counterexample_trace_selection/models/sketch_lora` | Produced by `scripts/train_adapter.py`. |
| Training checkpoints | `/workspace/large_artifacts/qwen35_4b_active_counterexample_trace_selection/models/sketch_lora/checkpoint-*` | Intermediate checkpoints, if present. |

