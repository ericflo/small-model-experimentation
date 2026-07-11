# Source

- `gym/`: self-contained C53 procedural environments.
- `curriculum.py`: shared process schema, state-aware experts, and semantic
  uncertainty diagnostics.
- `rollout.py`: live lockstep collection with exact visible-state, expert, and
  sampled-token receipts.
- `io_utils.py`: deterministic configuration, split, hashing, and JSONL tools.
- `harness.py` / `vllm_runner.py`: pinned bulk inference path copied from the
  validated repository template and C53 harness.
