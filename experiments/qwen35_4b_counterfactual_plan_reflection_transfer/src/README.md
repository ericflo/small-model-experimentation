# Source

- `taskgen.py` constructs fresh, exact-depth three-primitive list, string, and
  register machines plus exact-depth-1/2 retention tasks. It keeps composition
  and behavioral signatures disjoint across train, calibration, qualification,
  and confirmation splits.
- `records.py` builds the four immutable training arms, shared optimizer schedule,
  exact Qwen thinking-channel targets, and final-Assistant-only loss masks.
- `analyze.py` implements the paired bootstrap, family-breadth, mechanism, positive
  control, and retention gates.
- `score_artifacts.py`, `gate_artifacts.py`, and `adapter_gate_artifacts.py` reconstruct
  score and gate evidence from hash-bound raw artifacts instead of trusting copied
  pass fields.
- `checkpoint_lineage.py` opens and inventories retained LoRA tensors and merged
  safetensors shards before a model override is accepted.
- `merge_replay.py` authenticates the exact pinned base shards and proves every merged
  tensor is either unchanged or the exact registered LoRA update.
- `tensor_merge.py` writes the exact pinned two-shard layout without instantiating the
  model, preserving every unchanged tensor's source dtype and applying LoRA only to
  the registered target weights.
- `vllm_runner.py` is the repository-pinned Qwen3.5-4B bulk-inference runner.
- `runtime_contract.py` and `tokenizer_lineage.py` bind every execution stage to one
  detached exact-SHA worktree and the complete tokenizer/runtime identity.
- `matched_compute.py` validates the outcome-blind frozen reservoir, dual-unit compute
  stop, raw generation ancestry, and two-seed final promotion comparison.

Only a fresh tokenizer receipt is authorized; the historical receipt predates the
current schema and cannot authorize training. QLoRA training, vLLM generation,
evaluation, and any future Transformers Jacobian work remain unauthorized until an
independent review opens the corresponding committed gates.
