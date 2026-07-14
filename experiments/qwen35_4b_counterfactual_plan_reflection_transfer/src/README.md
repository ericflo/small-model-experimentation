# Source

- `taskgen.py` constructs fresh, exact-depth three-primitive list, string, and
  register machines plus exact-depth-1/2 retention tasks. It keeps composition
  and behavioral signatures disjoint across train, calibration, qualification,
  and confirmation splits.
- `records.py` builds the four immutable training arms, shared optimizer schedule,
  exact Qwen thinking-channel targets, and final-Assistant-only loss masks.
- `analyze.py` implements the paired bootstrap, family-breadth, mechanism, positive
  control, and retention gates.
- `vllm_runner.py` is the repository-pinned Qwen3.5-4B bulk-inference runner.

The current path is deliberately model-free. The tokenizer receipt, QLoRA trainer,
vLLM generation, and any future Transformers Jacobian work remain unauthorized until
the corresponding committed gates open.
