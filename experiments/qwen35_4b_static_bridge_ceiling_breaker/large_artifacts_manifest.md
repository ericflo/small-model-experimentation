# Large Artifacts Manifest

Large model outputs are intentionally kept outside the compact experiment directory.

Root:

`/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/`

Adapter directories:

- `/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/models/seed_lora` (445M)
- `/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/models/static60_lora` (445M)
- `/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/models/static80_lora` (445M)

Each adapter directory contains:

- final adapter weights: `adapter_model.safetensors` (163M),
- tokenizer files,
- trainer metadata,
- checkpoint directory: `checkpoint-60` (264M, includes optimizer state and checkpoint adapter weights).

Total large model area:

`/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/models` (1.4G)

Compact experiment directory:

`/workspace/experiments/qwen35_4b_static_bridge_ceiling_breaker` (7.2M)

The compact directory contains datasets, scripts, reports, figures, logs, and metadata only.
