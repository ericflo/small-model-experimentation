# Large Artifacts Manifest

Large model outputs are intentionally kept outside the compact experiment directory.

Root:

`/workspace/large_artifacts/qwen35_4b_typed_sketch_synthesis/`

Adapter directories:

- `/workspace/large_artifacts/qwen35_4b_typed_sketch_synthesis/models/program_lora` (`445M`)
- `/workspace/large_artifacts/qwen35_4b_typed_sketch_synthesis/models/sketch_lora` (`445M`)

Large files:

- `models/program_lora/adapter_model.safetensors` (`~162M`)
- `models/program_lora/checkpoint-60/adapter_model.safetensors` (`~162M`)
- `models/sketch_lora/adapter_model.safetensors` (`~162M`)
- `models/sketch_lora/checkpoint-60/adapter_model.safetensors` (`~162M`)

The compact directory contains datasets, scripts, reports, figures, logs, and metadata only.
