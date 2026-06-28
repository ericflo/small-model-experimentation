# Qwen 3.5 4B Unsaturated Frontier Active Bridge

This standalone experiment tests active bridge allocation on a broad executable-DSL frontier suite.

The model receives:

- an input schema,
- a current wrong DSL program,
- visible execution cases,
- and must output one corrected DSL expression.

The experiment trains four fixed-budget adapters:

- `seed_lora`: 240 base-family random-trace records.
- `static_bridge_lora`: 180 base-family records plus 60 uniformly allocated static frontier bridge records.
- `seed_mined_bridge_lora`: 180 base-family records plus 60 uniformly allocated bridge records selected against seed-adapter wrong programs.
- `adaptive_bridge_lora`: 180 base-family records plus 60 bridge records adaptively allocated from wrong programs generated after static bridge training.

## Layout

- `configs/experiment.json`: fixed design and hyperparameters.
- `src/`: standalone DSL, data, prompt, and model utilities.
- `scripts/`: dataset generation, mining, training, evaluation, and report entry points.
- `data/`: generated JSONL datasets and manifests.
- `reports/`: mining JSON, evaluation JSON files, and final report.
- `logs/` and `run_logs/`: experiment notebook and command output.
- `large_artifacts_manifest.md`: pointers to adapter directories stored outside this compact directory.

## Main Readout

- Final report: `reports/qwen35_4b_unsaturated_frontier_active_bridge_report.md`.
- Seed adapter frontier reranked hidden all-pass: 62/120 = 51.7%.
- Static bridge adapter frontier reranked hidden all-pass: 118/120 = 98.3%.
- Seed-mined bridge adapter frontier reranked hidden all-pass: 101/120 = 84.2%.
- Adaptive bridge adapter frontier reranked hidden all-pass: 102/120 = 85.0%.
- All four adapters retained 60/60 hidden all-pass on the IID eval split.
- Adaptive trace controls: aligned trace 102/120, no trace 70/120, shuffled trace 21/120.

Large adapters and checkpoints are intentionally outside this directory:

`/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/`
