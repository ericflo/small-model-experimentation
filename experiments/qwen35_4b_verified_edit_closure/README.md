# Qwen 3.5 4B Verified Edit Closure

**Status:** finished

This standalone experiment tests whether Qwen 3.5 4B executable-DSL repair improves when model candidates are expanded by a bounded symbolic edit neighborhood and selected by visible execution.

The model receives:

- an input schema,
- a current wrong DSL program,
- visible execution cases with expected and got values,
- and must output one corrected executable DSL expression.

The experiment trains one fixed-budget adapter:

- `static60_lora`: 180 base-family records plus 60 support bridge records.

The inference-time edit closure starts from the adapter's generated candidates and enumerates local DSL edits such as primitive substitutions, missing negation, raw/sorted tuple access, modulo scope changes, and missing conjunction terms. Variants are executed on visible cases, and the best visible-passing variant is scored on hidden cases.

Large adapters and checkpoints are intentionally outside this compact directory:

`/workspace/large_artifacts/qwen35_4b_verified_edit_closure/`

## Layout

- `configs/experiment.json`: fixed design and hyperparameters.
- `src/`: standalone DSL, data, prompt, model, and edit-closure utilities.
- `scripts/`: dataset generation, training, baseline evaluation, edit-closure evaluation, and report entry points.
- `data/`: generated JSONL datasets and manifest.
- `reports/`: evaluation JSON files, final report, and generated charts under `reports/figures/`.
- `logs/` and `run_logs/`: experiment notebook and command output.
- `large_artifacts_manifest.md`: pointers to adapter directories stored outside this compact directory.

## Report

Final report:

`reports/qwen35_4b_verified_edit_closure_report.md`

## Final Results

- IID baseline rerank hidden all-pass: 60/60.
- Support baseline rerank hidden all-pass: 120/120.
- Ceiling baseline rerank hidden all-pass: 47/120.
- Ceiling visible-selected edit closure hidden all-pass: 62/120.
- Ceiling strict visible-all closure hidden all-pass: 61/120.
- Ceiling hidden-oracle closure all-pass: 69/120.
- Strict closure accepted 39/120 edits and had 4 hidden pass-count damage cases.
