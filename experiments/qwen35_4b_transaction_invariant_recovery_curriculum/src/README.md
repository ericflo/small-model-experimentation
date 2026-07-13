# Source

- `repo_tasks.py`: procedural transaction, prior recovery, and transfer
  repositories plus the constrained executable environment.
- `repo_agent.py`: real looping JSON-tool coding agent and controlled public
  recovery states.
- `bank.py`: executable full/partial replay, seven conditional transition rows,
  firewall checks, and action-mass calibration.
- `harness.py` / `vllm_runner.py`: the pinned merged-checkpoint vLLM backend.

This directory is self-contained. It never imports or reads benchmark family
modules; benchmark use is allowed only through the suite's public CLI after all
white-box gates pass.
