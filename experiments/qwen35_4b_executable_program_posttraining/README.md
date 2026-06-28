# Qwen 3.5 4B Executable Program Posttraining

This standalone experiment tests whether a small posttraining change can move Qwen 3.5 4B from direct text repair into executable program repair.

The model receives a wrong DSL program plus visible failing cases and emits a corrected DSL expression. The evaluator executes generated programs on visible cases, optionally reranks sampled candidates by visible pass count, and then scores hidden cases.

## Layout

- `configs/experiment.json`: fixed model, data, training, and evaluation settings.
- `src/`: standalone DSL, prompting, data, and model utilities.
- `scripts/`: dataset generation, training, evaluation, and reporting entry points.
- `data/`: generated JSONL train/eval records.
- `reports/`: metrics, per-record generations, and final writeup.
- `logs/` and `run_logs/`: experiment notes and command output.

Large generated artifacts are intentionally outside this directory:

- `/workspace/large_artifacts/qwen35_4b_executable_program_posttraining/models/`

The compact experiment directory can be downloaded without adapter weights or checkpoints.

## Intended Run

```bash
python scripts/build_dataset.py
python scripts/train_dsl_lora.py --mode trace --output-dir /workspace/large_artifacts/qwen35_4b_executable_program_posttraining/models/dsl_trace_lora
python scripts/eval_dsl.py --adapter /workspace/large_artifacts/qwen35_4b_executable_program_posttraining/models/dsl_trace_lora --prompt-mode trace --split holdout --output reports/eval_trace_holdout.json
python scripts/make_report.py
```
