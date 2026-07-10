# Source

- `vllm_runner.py` is the sole inference wrapper. This local schema-4 variant accepts explicit
  CUDA-graph capture sizes, passes them to vLLM, records the resolved compilation config, and aborts
  if vLLM changes the frozen list/maximum or does not enable full decode CUDA graphs.
- `model_harness.py` owns model-facing records, parsing, and sampled-token accounting. Every model
  generation path goes through the local vLLM runner.
- `macro_domain.py` owns the procedural verified-macro DSL and exact execution checks.
- `scientific_artifacts.py` owns live KV and CUDA-graph preflight validation, external storage,
  receipt-last commits, catalogs, state transitions, and K4 nonpromotion.

The sole permitted model is `Qwen/Qwen3.5-4B` at the pinned revision. No Transformers inference or
mixed-backend comparison is permitted.
