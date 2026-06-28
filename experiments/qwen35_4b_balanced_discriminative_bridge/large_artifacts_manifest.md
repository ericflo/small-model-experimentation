# Large Artifacts Manifest

Large model outputs are intentionally kept outside the compact experiment directory.

Root:

`/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/`

Adapter directories:

- `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/seed_lora` - 445M
- `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/static_bridge_lora` - 445M
- `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/alias_discriminative_bridge_lora` - 445M
- `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/model_discriminative_bridge_lora` - 445M

Total adapter directory size:

`/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models` - 1.8G

Compact experiment directory size at audit:

`/workspace/experiments/qwen35_4b_balanced_discriminative_bridge` - 13M

The compact directory contains datasets, scripts, reports, logs, and metadata only.

Audit: no files larger than 50M are present in the compact experiment directory.
