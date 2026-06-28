# Qwen3.5-4B Learned Active Trace Policy

Standalone experiment package for training and evaluating a learned active execution-query policy after typed sketch synthesis.

The experiment trains two local Qwen3.5-4B LoRA adapters:

- a sketch adapter that emits typed DSL sketches from visible traces;
- a policy adapter that chooses the next query input from candidate-output bucket summaries.

Large artifacts such as adapters and checkpoints are stored outside this directory under:

`/workspace/large_artifacts/qwen35_4b_learned_active_trace_policy`

## Directory Layout

- `configs/`: experiment configuration.
- `data/`: generated JSONL datasets and manifests.
- `logs/`: human-readable experiment log.
- `reports/`: result JSON, CSV summaries, figures, and final report.
- `run_logs/`: command stdout/stderr captures.
- `scripts/`: runnable experiment scripts.
- `src/`: local DSL, data-generation, sketch, prompt, modeling, and active-policy code.

## Main Commands

```bash
python scripts/build_dataset.py
python scripts/build_policy_dataset.py --train-data data/static_bridge_80/dsl_train.jsonl --eval-data data/seed/dsl_train.jsonl
python scripts/train_adapter.py --task sketch --target-field target_sketch ...
python scripts/train_policy_adapter.py --train data/policy/policy_train.jsonl --eval data/policy/policy_eval.jsonl ...
python scripts/eval_learned_policy.py --data data/eval/dsl_eval_ceiling.jsonl ...
python scripts/make_report.py
```
