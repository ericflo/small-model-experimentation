# Experiment Log

## 2026-06-22

- Created standalone experiment directory.
- Selected `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Defined the central question: whether active bridge allocation can outperform uniform static bridge coverage on a broad executable-DSL frontier suite.
- Fixed the adapter training budget at 240 records per condition.
- Planned conditions:
  - seed adapter: 240 base-family random-trace records,
  - static bridge adapter: 180 base-family records plus 60 uniformly allocated static frontier bridge records,
  - seed-mined bridge adapter: 180 base-family records plus 60 uniformly allocated bridge records selected against seed-adapter wrong programs,
  - adaptive bridge adapter: 180 base-family records plus 60 bridge records allocated toward wrong programs still produced after static bridge training.
- Frontier suite:
  - 10 held-out compositional families,
  - 12 eval records per family,
  - 24 mining records per family,
  - 6 visible cases and 18 hidden cases per ordinary record.
- Large model artifacts will be stored under `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/`.

Next step: build and validate the datasets.

### Dataset Build

- Ran `python scripts/build_dataset.py`.
- Wrote seed train split: 240 records.
- Wrote static bridge train split: 240 records.
- Wrote base anchor split for mined/adaptive bridge training: 180 records.
- Wrote static frontier bridge records: 60 records.
- Wrote IID eval split: 60 records.
- Wrote frontier eval split: 120 records, 12 per frontier family.
- Wrote mining pool: 240 records, 24 per frontier family.
- Each ordinary record has 6 visible cases and 18 hidden cases.
- Each mining record has a 96-case pool for identifying executable wrong programs.
- Static bridge allocation: 6 records for each of the 10 frontier families.
- Dataset manifest: `data/dataset_manifest.json`.

Next step: train the seed and static bridge adapters, then use them for fixed and adaptive mining.

### Seed Adapter Training

- Trained `seed_lora` on `data/seed/dsl_train.jsonl`.
- Training records: 240.
- Eval records used during training: first 24 IID records from `data/eval/dsl_eval_iid.jsonl`.
- Epochs: 2.
- LoRA rank/alpha/dropout: 32/64/0.05.
- Final IID eval loss: 0.0001375.
- Training runtime: about 855 seconds.
- Train loss: about 0.1087.
- Adapter output: `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/seed_lora`.

Next step: train the uniform static bridge adapter under the same record budget and hyperparameters.

### Static Bridge Adapter Training

- Trained `static_bridge_lora` on `data/static_bridge/dsl_train.jsonl`.
- Training records: 240.
- Composition:
  - 180 base-family records,
  - 60 frontier-family static bridge records,
  - 6 bridge records for each frontier family.
- Epochs: 2.
- Final IID eval loss: 0.0005135.
- Training runtime: about 893 seconds.
- Train loss: about 0.1119.
- Adapter output: `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/static_bridge_lora`.

Next step: mine executable wrong programs from the seed adapter and build the seed-mined bridge training split.

### Seed-Adapter Mining

- First mining pass exposed a data-quality issue: syntactically parseable but non-executable candidate programs could be counted as wrong programs.
- Patched `scripts/mine_model_counterexamples.py` so a candidate must parse and execute on every case-pool input before it can count as a wrong program for bridge selection.
- Removed stale seed-mined outputs and reran mining with the strict executable filter.
- Ran strict seed mining with:
  - adapter: `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/seed_lora`,
  - mining records: 240,
  - candidates per record: 1 greedy + 2 sampled,
  - max new tokens: 64,
  - allocation mode: fixed,
  - trace strategy: `seed_mined`.
- Wrote mining report: `reports/mining/seed_mined_mining.json`.
- Wrote seed-mined bridge records: `data/seed_mined/bridge_records.jsonl`.
- Wrote seed-mined train split: `data/seed_mined/dsl_train.jsonl`.
- Seed-mined train split:
  - 240 records total,
  - 180 base-family records,
  - 60 bridge records.
- Fixed bridge allocation remained 6 records per frontier family.
- Strict executable model-wrong record counts:
  - `contains_count_length_code`: 6/6 selected records had executable model wrong programs,
  - `length_contains_code`: 6/6,
  - `length_mod_contains_code`: 6/6,
  - `modulo_sum_label`: 0/6, fallback static,
  - `not_contains_length_code`: 6/6,
  - `sorted_index_offset_label`: 6/6,
  - `sum_length_branch_label`: 0/6, fallback static,
  - `sum_offset_mod_label`: 6/6,
  - `tuple_branch_label`: 1/6,
  - `tuple_sum_gate_label`: 1/6.
- Dominant strict executable wrong-program clusters:
  - count-vs-length substitutions in string and token tasks,
  - use of unsorted values in the sorted-index task,
  - adding offset after modulo rather than applying modulo after offset,
  - `or` instead of `and` in tuple-sum gate.

Next step: train the seed-mined bridge adapter under the same 240-record budget.

### Seed-Mined Bridge Adapter Training

- Trained `seed_mined_bridge_lora` on `data/seed_mined/dsl_train.jsonl`.
- Training records: 240.
- Composition:
  - 180 base-family records,
  - 60 frontier-family bridge records selected from strict executable seed-adapter wrong programs or static fallback where no executable wrong program was available.
- Epochs: 2.
- Final IID eval loss: 0.00008227.
- Training runtime: about 853 seconds.
- Train loss: about 0.1110.
- Adapter output: `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/seed_mined_bridge_lora`.

Next step: mine residual executable wrong programs from the static bridge adapter with adaptive family allocation.

### Static-Adapter Adaptive Mining

- Ran adaptive mining with:
  - adapter: `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/static_bridge_lora`,
  - mining records: 240,
  - candidates per record: 1 greedy + 2 sampled,
  - max new tokens: 64,
  - allocation mode: adaptive,
  - trace strategy: `adaptive_mined`.
- Wrote mining report: `reports/mining/adaptive_mining.json`.
- Wrote adaptive bridge records: `data/adaptive/bridge_records.jsonl`.
- Wrote adaptive train split: `data/adaptive/dsl_train.jsonl`.
- Adaptive train split:
  - 240 records total,
  - 180 base-family records,
  - 60 bridge records.
- Adaptive bridge allocation:
  - `sorted_index_offset_label`: 24 records,
  - `length_mod_contains_code`: 10 records,
  - `length_contains_code`: 9 records,
  - `contains_count_length_code`: 5 records,
  - `modulo_sum_label`: 2 records,
  - `not_contains_length_code`: 2 records,
  - `sum_length_branch_label`: 2 records,
  - `sum_offset_mod_label`: 2 records,
  - `tuple_branch_label`: 2 records,
  - `tuple_sum_gate_label`: 2 records.
- Trace strategy counts:
  - 48 records used `adaptive_mined`,
  - 12 records used `adaptive_mined_fallback_static`.
- Strict executable model-wrong record counts:
  - `sorted_index_offset_label`: 4/24 selected records had executable static-adapter wrong programs; wrong score 8; unique wrong program count 1,
  - `length_mod_contains_code`: 1/10; wrong score 2; unique wrong program count 1,
  - `length_contains_code`: 1/9; wrong score 2; unique wrong program count 1,
  - `contains_count_length_code`: 1/5; wrong score 1; unique wrong program count 1,
  - all other selected families: 0 executable static-adapter wrong programs, static fallback.
- Dominant residual executable wrong-program cluster: `sorted_index_offset_label` collapsed to `(format "SI{}" (add (sum values) offset))`, replacing sorted indexed extraction with a sum-based shortcut.

Next step: train the adaptive bridge adapter under the same 240-record budget.

### Adaptive Bridge Adapter Training

- Trained `adaptive_bridge_lora` on `data/adaptive/dsl_train.jsonl`.
- Training records: 240.
- Composition:
  - 180 base-family records,
  - 60 frontier-family bridge records allocated by residual executable wrong-program mining from the static bridge adapter.
- Epochs: 2.
- Final IID eval loss: 0.0005400.
- Training runtime: about 845 seconds.
- Train loss: about 0.1192.
- Adapter output: `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/adaptive_bridge_lora`.

Next step: run frontier executable evaluations for seed, static bridge, seed-mined bridge, and adaptive bridge adapters; then run trace controls and IID retention checks.

### Frontier Evaluations

- Ran matched frontier evaluations on `data/eval/dsl_eval_challenge.jsonl` with trace prompts, 3 sampled candidates, max new tokens 64.
- Seed adapter:
  - greedy hidden all-pass: 56/120 = 46.7%,
  - reranked hidden all-pass: 62/120 = 51.7%.
- Static bridge adapter:
  - greedy hidden all-pass: 119/120 = 99.2%,
  - reranked hidden all-pass: 118/120 = 98.3%.
- Seed-mined bridge adapter:
  - greedy hidden all-pass: 92/120 = 76.7%,
  - reranked hidden all-pass: 101/120 = 84.2%.
- Adaptive bridge adapter:
  - greedy hidden all-pass: 102/120 = 85.0%,
  - reranked hidden all-pass: 102/120 = 85.0%.
- Static bridge was the strongest condition by a wide margin. The adaptive bridge allocation repaired `sorted_index_offset_label` but lost full coverage on `not_contains_length_code`, `length_contains_code`, `contains_count_length_code`, and `sum_offset_mod_label`.

### Adaptive Trace Controls

- Ran greedy-only adaptive controls on `data/eval/dsl_eval_challenge.jsonl`.
- Aligned trace result from the main adaptive run:
  - greedy hidden all-pass: 102/120 = 85.0%.
- No-trace control:
  - greedy hidden all-pass: 70/120 = 58.3%.
- Shuffled-trace control:
  - greedy hidden all-pass: 21/120 = 17.5%.
- Interpretation: aligned trace content carries substantial task information. Shuffled traces are actively harmful, so the effect is not just extra prompt length or generic in-context formatting.

### IID Retention Evaluations

- Ran greedy-only IID evaluations on `data/eval/dsl_eval_iid.jsonl`.
- Seed adapter: 60/60 = 100.0% hidden all-pass.
- Static bridge adapter: 60/60 = 100.0% hidden all-pass.
- Seed-mined bridge adapter: 60/60 = 100.0% hidden all-pass.
- Adaptive bridge adapter: 60/60 = 100.0% hidden all-pass.
- No adapter showed measurable IID retention loss on this 60-record IID eval split.

### Report

- Generated final standalone report: `reports/qwen35_4b_unsaturated_frontier_active_bridge_report.md`.
- Key conclusion: on this frontier suite, uniformly allocated static bridge records were more impactful than either seed-model mined bridges or adaptive residual mining. The strongest next direction is not more adaptive allocation on the current residual-mining rule; it is to preserve broad bridge coverage while improving bridge record construction and reranking robustness.
