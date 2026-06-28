# Large Artifacts Manifest

Large files for this experiment are intentionally stored outside the compact experiment directory.

## Compact Directory

- Path: `experiments/targeted_bridge_allocation/`
- Size after report generation: 24 MB.
- Contents: scripts, configs, datasets, logs, JSON results, CSV summaries, Markdown reports, and figures.
- Verified absent file extensions: `.safetensors`, `.bin`, `.pt`, `.pth`.

## Large Artifact Directory

- Path: `large_artifacts/targeted_bridge_allocation/`
- Size after training and evaluation: 13 GB.
- Model root: `large_artifacts/targeted_bridge_allocation/models/`
- Adapter directories: 10.
- Files under model root: 410.
- Checkpoint directories: 30.

Adapter directories:

- `easy_target_control_trace_lora`
- `hard_target_no_trace_lora`
- `hard_target_seen_preserving_trace_lora`
- `hard_target_shuffled_trace_lora`
- `hard_target_trace_lora`
- `length16_trace_lora`
- `modulo16_trace_lora`
- `tuple16_trace_lora`
- `uniform2_trace_lora`
- `uniform4_trace_lora`
