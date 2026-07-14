# Source

- `protocol.py`: model-free first-stop and strict pre-commit contract.
- `vllm_runner.py`: immutable calibration runner and repository vLLM template.
- `mechanics_protocol.py`: canonical semantic program IDs, frozen selector,
  execution controls, and hidden exact-score primitives.
- `mechanics_runtime.py`: winner-bound tokenizer-EOS generation and exact
  request/output authentication without modifying the calibration runner.
- `mechanics_stage.py`: transport gate, suffix/direct mechanics generation,
  visible analysis, authenticated hidden decrypt, and terminal inference.
- `mechanics_lock.py`: exact-commit review, CI, release, frozen-blob, runtime,
  and preflight authorization checks for conditional mechanics.
- `plans.py`: frozen sampling and resource-matching plan helpers.
- `stats.py`: exact paired inference and interval helpers.

Calibration is complete. Mechanics remains unauthorized until a clean exact-
commit adversarial review passes and its receipt plus the resulting mechanics
lock are separately committed, pushed, and green.
