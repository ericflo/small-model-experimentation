# Experiment-local source

- `repo_tasks.py`: ten deterministic repository families and the constrained real filesystem/test environment.
- `repo_agent.py`: batched iterative tool loop with visible+private terminal grading and operator-retention receipts.
- `bank.py`: successful-patch minimization, per-file collapse, canonical executable replay, firewall checks, and exact tokenizer-level action/plan mass calibration.
- `harness.py` / `vllm_runner.py`: pinned Qwen3.5-4B vLLM generation with merged-checkpoint support and an exact architecture fingerprint.

Hidden executable source and host oracle patches must never be serialized by this package. `bank.assert_firewall_clean` is mandatory before writing a harvest or bank.
