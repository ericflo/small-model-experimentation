# Qwen 3.5 4B Typed Sketch Synthesis Log

## Objective

Test whether typed partial-program synthesis can improve held-out executable DSL repair over direct program generation for Qwen 3.5 4B.

## Design Commitments

- Use only `Qwen/Qwen3.5-4B`.
- Train fresh adapters inside this standalone experiment.
- Keep the training budget fixed at 240 records per adapter.
- Keep adapter/checkpoint files outside the compact experiment directory.
- Evaluate direct program generation and typed-sketch synthesis on IID, support, and held-out ceiling splits.
- Report both visible-selected hidden success and hidden-oracle synthesis coverage.
- Generate a final markdown report and charts.

## Hypotheses

1. Some held-out failures require compositional jumps that local edits cannot generate.
2. A model-generated typed sketch can provide enough structure for bounded symbolic completion to find those jumps.
3. If hidden-oracle synthesis coverage is much higher than visible-selected synthesis success, the bottleneck is visible-case discrimination.
4. If hidden-oracle synthesis coverage remains low, the sketch space or expression bank is still not expressive enough.

## Planned Runs

1. Build deterministic datasets from seed `20260701`.
2. Add deterministic target sketches to every record.
3. Train `program_lora` on complete DSL programs.
4. Train `sketch_lora` on typed DSL sketches.
5. Evaluate `program_lora` on IID, support, and ceiling splits.
6. Evaluate typed-sketch synthesis on IID, support, and ceiling splits.
7. Iterate the synthesizer or selector if early checks expose obvious failures.
8. Generate charts and final report.
9. Audit compact artifact size and large artifact separation.

## Step Log

- Initialized standalone experiment directory and large artifact directory.
- Copied stable DSL, data generation, prompt, training, and direct-program evaluator utilities.
- Implemented program-vs-sketch training targets, typed sketch prompts, deterministic target sketch generation, bounded typed synthesis, and sketch evaluation.
- Built datasets with seed `20260701`.
- Target sketch recovery preflight:
  - `data/static_bridge_60/dsl_train.jsonl`: 240/240 target programs recovered.
  - `data/eval/dsl_eval_iid.jsonl`: 60/60 target programs recovered.
  - `data/eval/dsl_eval_support.jsonl`: 120/120 target programs recovered.
  - `data/eval/dsl_eval_ceiling.jsonl`: 120/120 target programs recovered.
- Recovery preflight finding: initial ranking over-prioritized label-length numeric features and under-prioritized literal `0` for scalar gates. Fixed ranking before any model training.
- Trained `program_lora` in `/workspace/large_artifacts/qwen35_4b_typed_sketch_synthesis/models/program_lora`.
  - Final eval loss: `0.0001638`.
  - Train runtime: `857.4` seconds.
- Trained `sketch_lora` in `/workspace/large_artifacts/qwen35_4b_typed_sketch_synthesis/models/sketch_lora`.
  - Final eval loss: `0.000158`.
  - Train runtime: `916.7` seconds.
- Direct program evaluation:
  - IID: 60/60 hidden all-cases success.
  - Support: 117/120 hidden all-cases success.
  - Ceiling: 40/120 hidden all-cases success.
- Initial sketch smoke evaluation on five ceiling records failed: 0/5 target synthesized and 0/5 hidden-oracle success.
- Iteration: added structural sketch abstraction variants and deeper typed expression-bank entries.
  - Five-record smoke improved to 5/5 target synthesized, 5/5 hidden-oracle success, and 2/5 selected hidden success.
- Iteration: changed visible-pass ties to prefer input-dependent and structurally richer candidates.
  - Five-record smoke improved to 5/5 target synthesized, 5/5 hidden-oracle success, and 4/5 selected hidden success.
- Iteration: reordered abstraction variants first and evaluated greedy-only sketch generation with an 8,000 total-candidate cap.
  - Five-record smoke retained 5/5 target synthesized, 5/5 hidden-oracle success, and 4/5 selected hidden success.
- Iteration: fixed candidate tag merging so targeted predicate entries can promote generic candidates already present in the expression bank.
  - This fixed `sorted_join_contains_code` synthesis.
- Final sketch evaluation with greedy sketch generation and 8,000 total-candidate cap:
  - IID: 45/60 sketch-selected, 60/60 sketch-oracle, 60/60 conservative hybrid.
  - Support: 70/120 sketch-selected, 120/120 sketch-oracle, 118/120 conservative hybrid.
  - Ceiling: 94/120 sketch-selected, 120/120 sketch-oracle, 106/120 conservative hybrid.
- Generated final report and figures under `reports/`.
