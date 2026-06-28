# Large Artifacts Manifest

Large model artifacts for this experiment are intentionally stored outside this compact experiment directory.

## Large Artifact Root

`/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge`

## Adapter Directories

| Artifact | Path | Approx Size |
| --- | --- | ---: |
| Seed adapter | `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/seed_lora` | 445M |
| Static bridge adapter | `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/static_bridge_lora` | 445M |
| Seed-mined bridge adapter | `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/seed_mined_bridge_lora` | 445M |
| Adaptive bridge adapter | `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/adaptive_bridge_lora` | 445M |

## Notes

- These adapter directories contain LoRA weights, tokenizer files, adapter configs, and training metadata.
- The compact experiment directory contains source, configs, generated datasets, mining reports, evaluation reports, logs, and this manifest.
- The compact experiment directory does not contain checkpoint or adapter weight files.
