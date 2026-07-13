# Qwen3.5-4B Active Counterexample Trace Selection

**Status:** finished

Standalone experiment package for testing active execution-case acquisition after typed sketch synthesis.

The experiment trains a Qwen3.5-4B LoRA to emit typed DSL sketches, synthesizes executable candidate programs from those sketches, and compares selection policies that either commit from the original visible trace or request additional counterexample traces from a per-record query pool.

Large artifacts such as LoRA adapters and checkpoints are stored outside this directory under:

`/workspace/large_artifacts/qwen35_4b_active_counterexample_trace_selection`

## Directory Layout

- `configs/`: experiment configuration.
- `data/`: generated JSONL datasets and manifests.
- `logs/`: human-readable experiment log.
- `reports/`: result JSON, figures, and final report.
- `run_logs/`: command stdout/stderr captures.
- `scripts/`: runnable experiment scripts.
- `src/`: local DSL, data-generation, sketch, prompt, and modeling code.

## Main Commands

```bash
python scripts/build_dataset.py
python scripts/train_adapter.py --task sketch --target-field target_sketch ...
python scripts/eval_active_selection.py --data data/eval/dsl_eval_ceiling.jsonl ...
python scripts/make_report.py
```
