# Experiment-local source

- `vllm_runner.py` is the immutable high-throughput inference wrapper copied byte-for-byte from the
  direct parent. Its required SHA-256 is
  `fd9972bdcb3a9e8b9841b45ed8e2849017a6e80b601e924817cdaaa5144b8782`; do not edit it in this
  experiment.
- `model_harness.py` builds the frozen prompts and delegates the complete batch to that vLLM runner.
- `macro_domain.py` is the frozen procedural DSL executor and verifier.
- `scientific_artifacts.py` owns the capacity-fit storage protocol: safe external paths, fixed
  geometry, preflight validation, receipt-last commits, protocol/runtime binding, full-tree catalogs,
  first-adequate selection validation, and K4 nonpromotion.

All four files are bound into each receipt's protocol identity. Model-facing work is allowed only
with `Qwen/Qwen3.5-4B` at the pinned revision; no Transformers fallback exists.
