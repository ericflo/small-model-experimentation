# Source

- `repo_tasks.py` and `repo_agent.py` implement the held-out procedural coding
  repositories, constrained tools, and batched iterative agent loop.
- `gym/`, `tasks.py`, `gen_tasks.py`, `code_env.py`, and `harness.py` are copied
  from round 1 for a self-contained same-substrate mechanism evaluation.
- `vllm_runner.py` is the round-1 runner with merged-checkpoint
  `model_override` support, required by the C49 deployment path.
- `loopdetect.py` provides the unchanged termination guard.
