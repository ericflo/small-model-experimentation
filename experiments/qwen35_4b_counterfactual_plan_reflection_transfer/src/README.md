# Source

- `taskgen.py` constructs fresh, exact-depth three-primitive list, string, and
  register machines. It keeps composition and behavioral signatures disjoint
  across train, qualification, and confirmation splits.
- `vllm_runner.py` is the repository-pinned Qwen3.5-4B bulk-inference runner.

The current path is deliberately model-free. Training, vLLM generation, and
Transformers Jacobian code remain unauthorized until design review passes.
