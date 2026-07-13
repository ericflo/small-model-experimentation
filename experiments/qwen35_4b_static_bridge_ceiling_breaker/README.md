# Qwen 3.5 4B Static Bridge Ceiling Breaker

**Status:** finished

This standalone experiment tests whether static bridge posttraining learns a trace-conditioned executable repair interface that transfers from support bridge families to deeper held-out composition families.

The model receives:

- an input schema,
- a current wrong DSL program,
- visible execution cases with expected and got values,
- and must output one corrected executable DSL expression.

The trained conditions are:

- `seed_lora`: 240 base-family random-trace records.
- `static60_lora`: 180 base-family records plus 60 equal support bridge records.
- `static80_lora`: 160 base-family records plus 80 equal support bridge records.

The main evaluation split is `dsl_eval_ceiling.jsonl`, whose families are absent from bridge training.

## Layout

- `configs/experiment.json`: fixed design and hyperparameters.
- `src/`: standalone DSL, data, prompt, and model utilities.
- `scripts/`: dataset generation, training, evaluation, and report entry points.
- `data/`: generated JSONL datasets and manifest.
- `reports/`: evaluation JSON files and final report.
- `figures/`: generated charts.
- `logs/` and `run_logs/`: experiment notebook and command output.
- `large_artifacts_manifest.md`: pointers to adapter directories stored outside this compact directory.

## Report

Final report:

`reports/qwen35_4b_static_bridge_ceiling_breaker_report.md`

Key results:

- Support reranked hidden all-pass: seed 64/120, Static60 120/120, Static80 120/120.
- Ceiling reranked hidden all-pass: seed 24/120, Static60 53/120, Static80 49/120.
- IID retention: all three adapters 60/60.
- Static60 trace controls on ceiling, greedy hidden: aligned 46/120, no trace 18/120, shuffled trace 8/120.

Large adapters and checkpoints are intentionally outside this directory:

`/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/`
