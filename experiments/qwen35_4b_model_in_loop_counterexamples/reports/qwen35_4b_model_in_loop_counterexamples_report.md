# Qwen 3.5 4B Model-In-Loop Counterexamples

## Question

Can counterexamples selected against Qwen-generated wrong DSL programs improve executable program repair beyond static counterexample traces under the same training budget?

## Design

- Base model: `Qwen/Qwen3.5-4B`.
- Model output: one executable DSL expression.
- Training: 4-bit NF4 QLoRA adapters.
- Training budget: 240 records per trained adapter.
- Seed adapter: 240 base-family random-trace records.
- Static bridge adapter: 180 base-family records plus 60 challenge-family static counterexample records.
- Model-loop bridge adapter: 180 base-family records plus 60 challenge-family records whose traces were selected against seed-adapter wrong programs.
- Evaluation: parse and execute generated programs on visible and hidden cases.
- Candidate selection: choose the valid candidate with the most visible-case passes.
- Large adapter/checkpoint files are stored outside the compact experiment directory.

## Dataset

- Seed train records: 240.
- Static bridge train records: 240.
- Model-loop bridge allocation: {'length_contains_code': 40, 'modulo_sum_label': 10, 'tuple_branch_label': 10}.
- IID eval records: 60.
- Challenge eval records: 72.
- Mining pool records: 144.
- Visible cases per record: 6.
- Hidden cases per record: 18.

## Mining Summary

- `length_contains_code`: 36/40 bridge records had sampled model wrong programs; 2 unique wrong programs mined. Top: 105x `(if (and (contains text needle) (gt (count_eq text needle) threshold)) "MATCH_LONG" "MISS")`; 1x `(if (and (contains text needle) (gt (len (first (match-all text needle))) threshold)) "MATCH_LONG" "MISS")`.
- `modulo_sum_label`: 0/10 bridge records had sampled model wrong programs; 0 unique wrong programs mined. Top: none.
- `tuple_branch_label`: 10/10 bridge records had sampled model wrong programs; 3 unique wrong programs mined. Top: 14x `(if (and (gt (sum item) threshold) (gt (tuple_get item index) 0)) high_label low_label)`; 6x `(if (and (gt (sum item) threshold) (gt (tuple_get item index) threshold)) high_label low_label)`; 6x `(if (and (gt (tuple_get item index) threshold) high_label) low_label)`.

## Main Results

| Condition | Data | Prompt | Samples | Greedy Hidden | Rerank Hidden | modulo_sum_label | length_contains_code | tuple_branch_label |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed adapter | `dsl_eval_challenge.jsonl` | `trace` | 3 | 62.5% (45/72) | 63.9% (46/72) | 100.0% (24/24) | 25.0% (6/24) | 66.7% (16/24) |
| Static bridge adapter | `dsl_eval_challenge.jsonl` | `trace` | 3 | 100.0% (72/72) | 100.0% (72/72) | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) |
| Model-loop bridge adapter | `dsl_eval_challenge.jsonl` | `trace` | 3 | 98.6% (71/72) | 100.0% (72/72) | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) |
| Model-loop bridge adapter, no trace | `dsl_eval_challenge.jsonl` | `no_trace` | 0 | 86.1% (62/72) | 86.1% (62/72) | 100.0% (24/24) | 58.3% (14/24) | 100.0% (24/24) |
| Model-loop bridge adapter, shuffled trace | `dsl_eval_challenge.jsonl` | `shuffled_trace` | 0 | 36.1% (26/72) | 36.1% (26/72) | 45.8% (11/24) | 41.7% (10/24) | 20.8% (5/24) |
| Seed adapter, IID | `dsl_eval_iid.jsonl` | `trace` | 0 | 100.0% (60/60) | 100.0% (60/60) | n/a | n/a | n/a |
| Static bridge adapter, IID | `dsl_eval_iid.jsonl` | `trace` | 0 | 100.0% (60/60) | 100.0% (60/60) | n/a | n/a | n/a |
| Model-loop bridge adapter, IID | `dsl_eval_iid.jsonl` | `trace` | 0 | 100.0% (60/60) | 100.0% (60/60) | n/a | n/a | n/a |

## Readout

- Challenge reranked hidden all-pass: seed 63.9% (46/72), static bridge 100.0% (72/72), model-loop bridge 100.0% (72/72).
- `length_contains_code` reranked hidden all-pass: seed 25.0% (6/24), static bridge 100.0% (24/24), model-loop bridge 100.0% (24/24).
- `tuple_branch_label` reranked hidden all-pass: seed 66.7% (16/24), static bridge 100.0% (24/24), model-loop bridge 100.0% (24/24).
- `modulo_sum_label` reranked hidden all-pass: seed 100.0% (24/24), static bridge 100.0% (24/24), model-loop bridge 100.0% (24/24).
- Model-loop trace ablations: aligned trace 98.6% (71/72), no trace 86.1% (62/72), shuffled trace 36.1% (26/72).

## Failure Signatures

- Seed `length_contains_code` greedy programs: 18/24 `(if (and (contains text needle) (gt (count_eq text needle) threshold)) "MATCH_LONG" "MISS")`; 6/24 `(if (and (contains text needle) (gt (len text) threshold)) "MATCH_LONG" "MISS")`.
- Static bridge `length_contains_code` greedy programs: 24/24 `(if (and (contains text needle) (gt (len text) threshold)) "MATCH_LONG" "MISS")`.
- Model-loop bridge `length_contains_code` greedy programs: 24/24 `(if (and (contains text needle) (gt (len text) threshold)) "MATCH_LONG" "MISS")`.

## Interpretation

- Static bridge and model-loop bridge both solved the challenge set under reranking, while static bridge was cleaner under greedy decoding.
- Model-loop mining was still useful diagnostically: it exposed the seed adapter's stable wrong hypotheses, especially the `count_eq` substitute for string length.
- Trace ablations show the symbolic trace is semantically active. Removing it mainly hurts `length_contains_code`; shuffling it collapses all challenge families.
- For this task shape, the strongest training recipe is not yet the extra active-mining loop. It is targeted bridge coverage with an execution-based verifier.
- The next higher-leverage experiment should make bridge selection adaptive only after expanding the held-out challenge space enough that static bridge records no longer saturate it.

## Per-Condition Details

### seed_lora_challenge

- Adapter: `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/models/seed_lora`.
- Data: `data/eval/dsl_eval_challenge.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 62.5% (45/72).
- Rerank hidden all-pass: 63.9% (46/72).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| length_contains_code | 25.0% (6/24) | 25.0% (6/24) | 29.2% (7/24) | 29.2% (7/24) |
| modulo_sum_label | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) |
| tuple_branch_label | 62.5% (15/24) | 66.7% (16/24) | 66.7% (16/24) | 70.8% (17/24) |

### static_bridge_lora_challenge

- Adapter: `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/models/static_bridge_lora`.
- Data: `data/eval/dsl_eval_challenge.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 100.0% (72/72).
- Rerank hidden all-pass: 100.0% (72/72).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| length_contains_code | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) |
| modulo_sum_label | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) |
| tuple_branch_label | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) |

### model_loop_lora_challenge

- Adapter: `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/models/model_loop_lora`.
- Data: `data/eval/dsl_eval_challenge.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 98.6% (71/72).
- Rerank hidden all-pass: 100.0% (72/72).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| length_contains_code | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) |
| modulo_sum_label | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) |
| tuple_branch_label | 95.8% (23/24) | 100.0% (24/24) | 95.8% (23/24) | 100.0% (24/24) |

### model_loop_lora_no_trace_challenge

- Adapter: `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/models/model_loop_lora`.
- Data: `data/eval/dsl_eval_challenge.jsonl`.
- Prompt mode: `no_trace`.
- Samples: 0.
- Greedy hidden all-pass: 86.1% (62/72).
- Rerank hidden all-pass: 86.1% (62/72).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| length_contains_code | 58.3% (14/24) | 58.3% (14/24) | 58.3% (14/24) | 58.3% (14/24) |
| modulo_sum_label | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) |
| tuple_branch_label | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) |

### model_loop_lora_shuffled_trace_challenge

- Adapter: `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/models/model_loop_lora`.
- Data: `data/eval/dsl_eval_challenge.jsonl`.
- Prompt mode: `shuffled_trace`.
- Samples: 0.
- Greedy hidden all-pass: 36.1% (26/72).
- Rerank hidden all-pass: 36.1% (26/72).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| length_contains_code | 41.7% (10/24) | 41.7% (10/24) | 41.7% (10/24) | 41.7% (10/24) |
| modulo_sum_label | 45.8% (11/24) | 45.8% (11/24) | 45.8% (11/24) | 45.8% (11/24) |
| tuple_branch_label | 20.8% (5/24) | 20.8% (5/24) | 20.8% (5/24) | 20.8% (5/24) |

### seed_lora_iid

- Adapter: `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/models/seed_lora`.
- Data: `data/eval/dsl_eval_iid.jsonl`.
- Prompt mode: `trace`.
- Samples: 0.
- Greedy hidden all-pass: 100.0% (60/60).
- Rerank hidden all-pass: 100.0% (60/60).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_and_count_code | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| contains_code | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| contains_count_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| length_and_mod_code | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| length_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| length_mod_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| mod_scalar_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| scalar_branch_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| sorted_first_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| sum_add_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| sum_and_scalar_code | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| sum_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| sum_threshold_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| tuple_get_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| tuple_sum_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |

### static_bridge_lora_iid

- Adapter: `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/models/static_bridge_lora`.
- Data: `data/eval/dsl_eval_iid.jsonl`.
- Prompt mode: `trace`.
- Samples: 0.
- Greedy hidden all-pass: 100.0% (60/60).
- Rerank hidden all-pass: 100.0% (60/60).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_and_count_code | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| contains_code | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| contains_count_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| length_and_mod_code | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| length_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| length_mod_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| mod_scalar_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| scalar_branch_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| sorted_first_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| sum_add_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| sum_and_scalar_code | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| sum_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| sum_threshold_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| tuple_get_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| tuple_sum_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |

### model_loop_lora_iid

- Adapter: `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/models/model_loop_lora`.
- Data: `data/eval/dsl_eval_iid.jsonl`.
- Prompt mode: `trace`.
- Samples: 0.
- Greedy hidden all-pass: 100.0% (60/60).
- Rerank hidden all-pass: 100.0% (60/60).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_and_count_code | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| contains_code | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| contains_count_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| length_and_mod_code | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| length_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| length_mod_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| mod_scalar_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| scalar_branch_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| sorted_first_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| sum_add_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| sum_and_scalar_code | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| sum_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| sum_threshold_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| tuple_get_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |
| tuple_sum_label | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) | 100.0% (4/4) |

## Artifact Layout

- Compact artifacts: `/workspace/experiments/qwen35_4b_model_in_loop_counterexamples/`.
- Large artifacts: `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/`.
- Dataset manifest: `data/dataset_manifest.json`.
- Mining report: `reports/mining/seed_model_mining.json`.
- Evaluation JSON files: `reports/eval/`.
