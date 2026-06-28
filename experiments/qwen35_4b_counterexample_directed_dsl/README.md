# Qwen 3.5 4B Counterexample-Directed DSL

This standalone experiment tests whether visible execution traces become more useful when their examples are chosen to distinguish the target executable program from plausible wrong programs.

The model receives:

- an input schema,
- a current wrong DSL program,
- visible execution cases,
- and must output one corrected DSL expression.

The evaluator parses and executes generated DSL candidates on visible cases, optionally reranks candidates by visible pass count, and scores hidden cases.

## Layout

- `configs/experiment.json`: fixed design and hyperparameters.
- `src/`: standalone DSL, data, prompt, and model utilities.
- `scripts/`: dataset generation, training, evaluation, and report entry points.
- `data/`: generated JSONL datasets and manifests.
- `reports/`: evaluation JSON files and final report.
- `logs/` and `run_logs/`: experiment notebook and command output.

Large adapters and checkpoints are intentionally outside this directory:

`/workspace/large_artifacts/qwen35_4b_counterexample_directed_dsl/`

