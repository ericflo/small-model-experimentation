# Qwen 3.5 4B GraphIR Self Repair

**Status:** finished

This standalone experiment tests whether Qwen 3.5 4B can improve held-out executable repair by configuring a typed register graph and then applying a verifier-guided graph repair step.

The model receives:

- an input schema,
- a current wrong program or candidate graph,
- visible execution cases with expected and got values,
- and must output either one DSL expression or a GraphIR register program.

The trained conditions are:

- `dsl_static60_lora`: DSL baseline trained on 180 base records plus 60 support bridge records.
- `graphir_construct_lora`: same source records, target is typed GraphIR.
- `graphir_repair_lora`: corrupted candidate GraphIR plus visible mismatches, target is corrected GraphIR.

Large adapters and checkpoints are intentionally outside this compact directory:

`/workspace/large_artifacts/qwen35_4b_graphir_self_repair/`

## Layout

- `configs/experiment.json`: fixed design and hyperparameters.
- `src/`: standalone DSL, GraphIR, data, prompt, and model utilities.
- `scripts/`: dataset generation, training, evaluation, and report entry points.
- `data/`: generated JSONL datasets and manifest.
- `reports/`: evaluation JSON files and final report.
- `figures/`: generated charts.
- `logs/` and `run_logs/`: experiment notebook and command output.
- `large_artifacts_manifest.md`: pointers to adapter directories stored outside this compact directory.

## Report

Final report:

`reports/qwen35_4b_graphir_self_repair_report.md`

## Result

The main ceiling result was negative for the GraphIR pipeline:

- DSL baseline rerank: 35/120 hidden all-pass.
- GraphIR construction: 26/120 hidden all-pass.
- GraphIR construction plus repair: 29/120 hidden all-pass.
- Direct synthetic corrupted-GraphIR repair diagnostic: 4/120 input hidden all-pass to 32/120 repaired hidden all-pass.

The repair adapter learned useful behavior on synthetic corrupted graphs and fixed the two GraphIR support failures, but actual construction errors on held-out ceiling families did not transfer well enough for GraphIR plus repair to beat the DSL baseline.
