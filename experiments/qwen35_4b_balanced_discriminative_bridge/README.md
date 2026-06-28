# Qwen 3.5 4B Balanced Discriminative Bridge

This standalone experiment tests whether equal frontier-family bridge coverage improves when visible traces are selected to discriminate against hard alias programs and seed-adapter mistakes.

The model receives:

- an input schema,
- a current wrong DSL program,
- visible execution cases with expected and got values,
- and must output one corrected executable DSL expression.

The experiment trains four fixed-budget adapters:

- `seed_lora`: 240 base-family random-trace records.
- `static_bridge_lora`: 180 base-family records plus 60 equally allocated normal frontier bridge records.
- `alias_discriminative_bridge_lora`: 180 base-family records plus 60 equally allocated hard-case frontier records selected against an expanded alias bank.
- `model_discriminative_bridge_lora`: 180 base-family records plus 60 equally allocated hard-case frontier records selected against seed-adapter wrong programs plus the alias bank.

## Layout

- `configs/experiment.json`: fixed design and hyperparameters.
- `src/`: standalone DSL, data, prompt, and model utilities.
- `scripts/`: dataset generation, mining, training, evaluation, and report entry points.
- `data/`: generated JSONL datasets and manifests.
- `reports/`: mining JSON, evaluation JSON files, and final report.
- `logs/` and `run_logs/`: experiment notebook and command output.
- `large_artifacts_manifest.md`: pointers to adapter directories stored outside this compact directory.

## Main Readout

The strongest condition was `static_bridge_lora`: 118/120 normal frontier hidden all-pass and 119/120 hard frontier hidden all-pass.

Hard-case discriminative trace selection did not improve this setting. `alias_discriminative_bridge_lora` reached 108/120 hard frontier rerank hidden all-pass, and `model_discriminative_bridge_lora` reached 99/120.

All four adapters retained 60/60 IID hidden all-pass. Trace controls for `static_bridge_lora` on the hard frontier were: correct trace 119/120, no trace 109/120, shuffled trace 19/120.

Final report:

`reports/qwen35_4b_balanced_discriminative_bridge_report.md`

Large adapters and checkpoints are intentionally outside this directory:

`/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/`
