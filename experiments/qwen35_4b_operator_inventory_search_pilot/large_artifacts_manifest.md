# Large Artifacts Manifest

No large artifacts are required for this no-training pilot.

If later iterations train Qwen3.5-4B adapters, they should be stored under:

`/workspace/large_artifacts/qwen35_4b_operator_inventory_search_pilot`

## Audit

- This experiment directory contains only source, config, logs, generated JSON/CSV summaries, a JSONL benchmark, and PNG figures.
- No checkpoints, adapters, model weights, caches, or tensor dumps are stored inside the experiment directory.
- The intended external large-artifact directory is separate from the downloadable experiment package.
- Final experiment directory size: 4.4 MB.
- Final external large-artifact directory size: 0.
