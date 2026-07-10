# Experiment-local source

- `vllm_runner.py`: the exact single-file vLLM wrapper used for every recorded model call. Its
  hash is bound into each runner artifact, so historical comments are left byte-frozen; amendments
  1 and 2 plus `docs/vllm_inference.md` contain the corrected Ada batch-invariance interpretation.
- `model_harness.py`: strict prompt builders, parsers, and token-accounting adapters around the
  runner.
- `macro_domain.py`: deterministic procedural task generation, exact DSL execution, behavioral
  depth checks, macro mining, and compression utilities inherited from the parent.
- `full_artifacts.py`: model-free canonical sharding, atomic-receipt, path-containment, checksum,
  and multi-shard summary validation shared by the full runner and analyzer. It never loads a
  model and is not an alternate inference wrapper.
- `scientific_artifacts.py`: model-free external storage, preflight-only/receipt-last transaction,
  path-containment, exact receipt, protocol binding, deterministic catalog, and logical smoke-tier
  selection validation. It never loads a model and never creates a repository-local promoted copy.

No source file may load or suggest another model, and no scientific arm may bypass
`vllm_runner.py`.
