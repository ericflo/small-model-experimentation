# Qwen 3.5 4B Typed Sketch Synthesis

This standalone experiment tests whether `Qwen/Qwen3.5-4B` can improve executable DSL repair by emitting a typed partial program that a bounded symbolic synthesizer completes and verifies on visible execution cases.

The experiment trains two fresh adapters under the same data budget:

- `program_lora`: emits one complete corrected DSL expression.
- `sketch_lora`: emits one typed DSL sketch with holes such as `?NUM0`, `?TEXT0`, and `?PRED0`.

The synthesizer fills sketch holes with type-valid expressions built from the input schema, executes each completed candidate on visible cases, and reports hidden-case success plus hidden-oracle coverage inside the candidate set.

Large adapters and checkpoints are intentionally outside this compact directory:

`/workspace/large_artifacts/qwen35_4b_typed_sketch_synthesis/`

## Layout

- `configs/experiment.json`: fixed design and hyperparameters.
- `src/`: standalone DSL, data, prompt, model, and typed-sketch utilities.
- `scripts/`: dataset generation, training, baseline evaluation, sketch-synthesis evaluation, and report entry points.
- `data/`: generated JSONL datasets and manifest.
- `reports/`: evaluation JSON files, final report, and generated charts under `reports/figures/`.
- `logs/` and `run_logs/`: experiment notebook and command output.
- `large_artifacts_manifest.md`: pointers to adapter directories stored outside this compact directory.

## Report

Final report path after evaluation:

`reports/qwen35_4b_typed_sketch_synthesis_report.md`
