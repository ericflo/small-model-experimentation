# Experiment Log

## 2026-06-22

- Created standalone experiment directory.
- Selected `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Defined the central question: whether counterexamples selected against model-generated wrong DSL programs improve executable repair beyond static counterexamples under the same training budget.
- Planned fixed-budget adapter conditions:
  - seed adapter: 240 base-family random-trace records,
  - static bridge adapter: 180 base-family records plus 60 challenge-family static counterexample records,
  - model-loop bridge adapter: 180 base-family records plus 60 challenge-family records selected against wrong programs mined from the seed adapter.
- Bridge allocation:
  - 40 `length_contains_code` records,
  - 10 `modulo_sum_label` records,
  - 10 `tuple_branch_label` records.
- Large model artifacts will be stored under `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/`.

Next step: build and validate the datasets, then train the seed adapter.

### Dataset Build

- Ran `python scripts/build_dataset.py`.
- Wrote seed train split: 240 records.
- Wrote static bridge train split: 240 records.
- Wrote base anchor split for model-loop training: 180 records.
- Wrote static challenge bridge records: 60 records.
- Wrote model mining pool: 144 records, 48 per challenge family.
- Wrote IID eval split: 60 records.
- Wrote challenge eval split: 72 records, 24 per challenge family.
- Each ordinary record has 6 visible cases and 18 hidden cases.
- Each mining-pool record has an additional 96-case pool for identifying model-generated wrong programs.
- Dataset manifest: `data/dataset_manifest.json`.

Next step: train the seed adapter on the 240-record base-family split.

### Seed Adapter Training

- Trained `seed_lora` on `data/seed/dsl_train.jsonl`.
- Training records: 240.
- Eval records used during training: first 24 IID records from `data/eval/dsl_eval_iid.jsonl`.
- Epochs: 2.
- LoRA rank/alpha/dropout: 32/64/0.05.
- Final IID eval loss: 0.0000395.
- Training runtime: about 839 seconds.
- Adapter output: `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/models/seed_lora`.

Next step: evaluate the seed adapter on the challenge split and mine model-generated wrong programs from the mining pool.

### Seed Challenge Evaluation

- Evaluated `seed_lora` on `data/eval/dsl_eval_challenge.jsonl`.
- Prompt mode: aligned trace.
- Samples: 3.
- Overall greedy hidden all-pass: 45/72.
- Overall reranked hidden all-pass: 46/72.
- Family results, reranked hidden all-pass:
  - `modulo_sum_label`: 24/24.
  - `length_contains_code`: 6/24.
  - `tuple_branch_label`: 16/24.

Next step: mine valid wrong programs from `seed_lora` on the separate mining pool and build the model-loop training split.

### Model-Candidate Mining

- Ran `scripts/mine_model_counterexamples.py` with `seed_lora`.
- Mining pool records: 144.
- Candidates per record: 1 greedy + 2 sampled.
- Wrote mining report: `reports/mining/seed_model_mining.json`.
- Wrote model-loop bridge records: `data/model_loop/model_mined_bridge_records.jsonl`.
- Wrote model-loop train split: `data/model_loop/dsl_train.jsonl`.
- Model-loop train split:
  - 240 records total,
  - 180 base-family records,
  - 60 bridge records.
- Mined bridge record composition:
  - 50 records with model-generated wrong programs,
  - 10 fallback static records for the solved modulo family.
- Mining summary:
  - `length_contains_code`: 36/40 requested bridge records had sampled model wrong programs; top wrong program occurred 105 times: `(if (and (contains text needle) (gt (count_eq text needle) threshold)) "MATCH_LONG" "MISS")`.
  - `tuple_branch_label`: 10/10 requested bridge records had sampled model wrong programs.
  - `modulo_sum_label`: 0/10 requested bridge records had sampled model wrong programs, so these use fallback static selectors.

Next step: train the static bridge and model-loop bridge adapters under the same 240-record budget.

### Static Bridge Adapter Training

- Trained `static_bridge_lora` on `data/static_bridge/dsl_train.jsonl`.
- Training records: 240.
- Composition:
  - 180 base-family records,
  - 60 challenge-family bridge records with static counterexample-selected traces.
- Epochs: 2.
- Final IID eval loss: 0.0000992.
- Training runtime: about 864 seconds.
- Adapter output: `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/models/static_bridge_lora`.

Next step: train the model-loop bridge adapter with the same training budget and hyperparameters.

### Model-Loop Bridge Adapter Training

- Trained `model_loop_lora` on `data/model_loop/dsl_train.jsonl`.
- Training records: 240.
- Composition:
  - 180 base-family records,
  - 50 challenge-family bridge records selected against model-generated wrong programs,
  - 10 challenge-family fallback static records for the solved modulo family.
- Epochs: 2.
- Final IID eval loss: 0.00003141.
- Training runtime: about 865 seconds.
- Adapter output: `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/models/model_loop_lora`.

Next step: run final challenge evaluations and prompt ablations.

### Final Challenge Evaluations

- Evaluated all adapters on `data/eval/dsl_eval_challenge.jsonl`.
- Evaluator behavior:
  - prompt mode: aligned trace,
  - samples: 3,
  - candidate selection: choose the valid candidate with the most visible-case passes,
  - success metric: generated DSL program must pass every hidden case for the record.
- `seed_lora`:
  - greedy hidden all-pass: 45/72,
  - reranked hidden all-pass: 46/72,
  - `length_contains_code`: 6/24 reranked hidden all-pass,
  - `modulo_sum_label`: 24/24 reranked hidden all-pass,
  - `tuple_branch_label`: 16/24 reranked hidden all-pass.
- `static_bridge_lora`:
  - greedy hidden all-pass: 72/72,
  - reranked hidden all-pass: 72/72,
  - all three challenge families: 24/24 reranked hidden all-pass.
- `model_loop_lora`:
  - greedy hidden all-pass: 71/72,
  - reranked hidden all-pass: 72/72,
  - `length_contains_code`: 24/24 reranked hidden all-pass,
  - `modulo_sum_label`: 24/24 reranked hidden all-pass,
  - `tuple_branch_label`: 24/24 reranked hidden all-pass.
- Raw result files:
  - `reports/eval/seed_lora_challenge.json`,
  - `reports/eval/static_bridge_lora_challenge.json`,
  - `reports/eval/model_loop_lora_challenge.json`.

### Prompt Controls

- Evaluated `model_loop_lora` on the same challenge split with greedy decoding only.
- No-trace control:
  - prompt mode: `no_trace`,
  - greedy hidden all-pass: 62/72,
  - `length_contains_code`: 14/24,
  - `modulo_sum_label`: 24/24,
  - `tuple_branch_label`: 24/24.
- Shuffled-trace control:
  - prompt mode: `shuffled_trace`,
  - greedy hidden all-pass: 26/72,
  - `length_contains_code`: 10/24,
  - `modulo_sum_label`: 11/24,
  - `tuple_branch_label`: 5/24.
- Interpretation:
  - aligned traces are semantically active,
  - removing traces mainly damages the length-vs-count distinction,
  - shuffling traces damages every challenge family.
- Raw result files:
  - `reports/eval/model_loop_lora_no_trace_challenge.json`,
  - `reports/eval/model_loop_lora_shuffled_trace_challenge.json`.

### IID Retention Checks

- Evaluated all adapters on `data/eval/dsl_eval_iid.jsonl`.
- Prompt mode: aligned trace.
- Samples: 0.
- Results:
  - `seed_lora`: 60/60 greedy hidden all-pass,
  - `static_bridge_lora`: 60/60 greedy hidden all-pass,
  - `model_loop_lora`: 60/60 greedy hidden all-pass.
- Raw result files:
  - `reports/eval/seed_lora_iid.json`,
  - `reports/eval/static_bridge_lora_iid.json`,
  - `reports/eval/model_loop_lora_iid.json`.

### Report

- Generated final report with `python scripts/make_report.py`.
- Report path: `reports/qwen35_4b_model_in_loop_counterexamples_report.md`.
- Main readout:
  - the seed adapter failed mainly on `length_contains_code`,
  - static bridge solved the challenge suite at 72/72 greedy and reranked hidden all-pass,
  - model-loop bridge solved the challenge suite at 72/72 after reranking and 71/72 greedy,
  - IID retention remained 60/60 for every adapter.
- Practical conclusion:
  - the best recipe demonstrated here is targeted bridge coverage plus execution-based verification,
  - model-in-loop mining is valuable as a diagnostic and selector, but it did not beat the simpler static bridge on this challenge suite,
  - the next higher-leverage run should expand the held-out challenge space until static bridge records no longer saturate it, then use active mining to allocate records to unsolved wrong-hypothesis clusters.

### Artifact Split

- Compact experiment directory: `/workspace/experiments/qwen35_4b_model_in_loop_counterexamples/`.
- Compact directory size after report generation: about 6.2 MB.
- Large artifact directory: `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/`.
- Large directory size after training: about 1.4 GB.
- Large files are LoRA adapter outputs and checkpoint snapshots.
- Large artifact manifest: `large_artifacts_manifest.md`.
