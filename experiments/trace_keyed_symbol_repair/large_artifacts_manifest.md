# Trace-Keyed Symbol Repair Large Artifacts

The downloadable experiment directory intentionally excludes model adapters and checkpoints.

- Small experiment directory: `/workspace/experiments/trace_keyed_symbol_repair`.
- Large artifact directory: `/workspace/large_artifacts/trace_keyed_symbol_repair`.

Large artifacts:
- `/workspace/large_artifacts/trace_keyed_symbol_repair/models/final_patch_lora`
- `/workspace/large_artifacts/trace_keyed_symbol_repair/models/no_trace_lora`
- `/workspace/large_artifacts/trace_keyed_symbol_repair/models/pilot_trace_lora`
- `/workspace/large_artifacts/trace_keyed_symbol_repair/models/shuffled_trace_lora`
- `/workspace/large_artifacts/trace_keyed_symbol_repair/models/trace_lora`

To reproduce evaluations, keep the large artifact directory at the path above or update adapter paths in `scripts/run_final_evaluations.py`.
