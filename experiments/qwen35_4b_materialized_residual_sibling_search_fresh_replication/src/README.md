# Source

- `identity.py`: versioned task/request identities and exact parent lineage.
- `task_data.py`, `plans.py`, `protocol.py`, `stats.py`: frozen procedural DSL,
  construction, prompts/parsers, and registered inference helpers.
- `mechanics.py`: model-free mechanics controls, scoring, and gate arithmetic;
  byte-identical to the scientific parent.
- `vllm_runner.py`: pinned same-backend inference with the distinct model and
  tokenizer EOS contract.

`scripts/run_mechanics.py` is the only mechanics orchestrator. Its prepared
transaction is reviewed but live execution remains lock-gated.
