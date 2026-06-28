# Qwen 3.5 4B Unsaturated Frontier Active Bridge

## Question

Can active bridge allocation outperform uniform static bridge coverage on a frontier suite broad enough that static bridge examples do not automatically saturate the target space?

## Design

- Base model: `Qwen/Qwen3.5-4B`.
- Model output: one executable DSL expression.
- Training: 4-bit NF4 QLoRA adapters.
- Training budget: 240 records per trained adapter.
- Seed adapter: 240 base-family random-trace records.
- Static bridge adapter: 180 base-family records plus 60 uniformly allocated static frontier bridge records.
- Seed-mined bridge adapter: 180 base-family records plus 60 uniformly allocated bridge records selected against seed-adapter wrong programs.
- Adaptive bridge adapter: 180 base-family records plus 60 bridge records allocated toward wrong programs generated after static bridge training.
- Evaluation: parse and execute generated programs on visible and hidden cases.
- Candidate selection: choose the valid candidate with the most visible-case passes.
- Large adapter/checkpoint files are stored outside the compact experiment directory.

## Dataset

- Seed train records: 240.
- Static bridge train records: 240.
- Bridge total: 60.
- Frontier families: 10.
- Frontier eval records: 120.
- IID eval records: 60.
- Mining pool records: 240.
- Visible cases per record: 6.
- Hidden cases per record: 18.

## Seed-Adapter Mining Summary

- Allocation mode: `fixed`.
- Bridge allocation: `{'contains_count_length_code': 6, 'length_contains_code': 6, 'length_mod_contains_code': 6, 'modulo_sum_label': 6, 'not_contains_length_code': 6, 'sorted_index_offset_label': 6, 'sum_length_branch_label': 6, 'sum_offset_mod_label': 6, 'tuple_branch_label': 6, 'tuple_sum_gate_label': 6}`.
- `contains_count_length_code`: 6/6 selected records had model-generated wrong programs; wrong-candidate score 20; 3 unique wrong programs. Top: 10x `(if (and (contains tokens needle) (gt (count_eq tokens needle) threshold)) "MANY_LONG" "MISS")`; 6x `(if (and (contains tokens needle) (and (gt (count_eq tokens needle) threshold) (gt (len needle) min_len))) "MANY_LONG" "MISS")`; 4x `(if (and (contains tokens needle) (gt (count_eq tokens needle) threshold)) "MANY_LONG" (if (gt (len tokens) min_len) "MANY_LONG" "MISS"))`.
- `length_contains_code`: 6/6 selected records had model-generated wrong programs; wrong-candidate score 67; 1 unique wrong programs. Top: 67x `(if (and (contains text needle) (gt (count_eq text needle) threshold)) "MATCH_LONG" "MISS")`.
- `length_mod_contains_code`: 6/6 selected records had model-generated wrong programs; wrong-candidate score 23; 4 unique wrong programs. Top: 19x `(if (and (contains text needle) (eq (mod (count_eq text needle) modulus) target)) "HIT_MOD" "MISS")`; 2x `(if (and (contains text needle) (gt (count_eq text needle) target)) "HIT_MOD" "MISS")`; 1x `(if (and (contains text needle) (gt (count_eq text needle) target)) (format "HIT_MOD" (mod (count_eq text needle) modulus)) "MISS")`.
- `modulo_sum_label`: 0/6 selected records had model-generated wrong programs; wrong-candidate score 0; 0 unique wrong programs. Top: none.
- `not_contains_length_code`: 6/6 selected records had model-generated wrong programs; wrong-candidate score 72; 2 unique wrong programs. Top: 69x `(if (and (contains text needle) (gt (count_eq text needle) threshold)) "ABSENT_LONG" "OTHER")`; 3x `(if (and (contains text needle) (gt (count_eq text needle) threshold)) "OTHER" "ABSENT_LONG")`.
- `sorted_index_offset_label`: 6/6 selected records had model-generated wrong programs; wrong-candidate score 55; 3 unique wrong programs. Top: 49x `(format "SI{}" (add (tuple_get values index) offset))`; 4x `(format "SI{}" (add (tuple_get values (mod index (len values))) offset))`; 2x `(format "SI{}" (sub (add (tuple_get values index) offset) 0))`.
- `sum_length_branch_label`: 0/6 selected records had model-generated wrong programs; wrong-candidate score 0; 0 unique wrong programs. Top: none.
- `sum_offset_mod_label`: 6/6 selected records had model-generated wrong programs; wrong-candidate score 36; 2 unique wrong programs. Top: 24x `(format "OM{}" (add (mod (sum values) modulus) offset))`; 12x `(format "OM{}" (add (sum values) offset))`.
- `tuple_branch_label`: 1/6 selected records had model-generated wrong programs; wrong-candidate score 1; 1 unique wrong programs. Top: 1x `(if (if (gt (tuple_get item index) threshold) high_label low_label) high_label low_label)`.
- `tuple_sum_gate_label`: 1/6 selected records had model-generated wrong programs; wrong-candidate score 1; 1 unique wrong programs. Top: 1x `(if (or (gt (tuple_get item index) threshold) (gt (sum item) sum_threshold)) high_label low_label)`.

## Static-Adapter Adaptive Mining Summary

- Allocation mode: `adaptive`.
- Bridge allocation: `{'contains_count_length_code': 5, 'length_contains_code': 9, 'length_mod_contains_code': 10, 'modulo_sum_label': 2, 'not_contains_length_code': 2, 'sorted_index_offset_label': 24, 'sum_length_branch_label': 2, 'sum_offset_mod_label': 2, 'tuple_branch_label': 2, 'tuple_sum_gate_label': 2}`.
- `contains_count_length_code`: 1/5 selected records had model-generated wrong programs; wrong-candidate score 1; 1 unique wrong programs. Top: 1x `(if (and (contains tokens needle) (gt (count_eq tokens needle) threshold) (gt (len tokens) min_len)) "FOUND" "MISS")`.
- `length_contains_code`: 1/9 selected records had model-generated wrong programs; wrong-candidate score 2; 1 unique wrong programs. Top: 2x `(if (and (contains text needle) (gt (count_eq text needle) threshold)) "MATCH_LONG" "MISS")`.
- `length_mod_contains_code`: 1/10 selected records had model-generated wrong programs; wrong-candidate score 2; 1 unique wrong programs. Top: 2x `(if (and (contains text needle) (eq (mod (len text) modulus) target)) "HIT" "MISS")`.
- `modulo_sum_label`: 0/2 selected records had model-generated wrong programs; wrong-candidate score 0; 0 unique wrong programs. Top: none.
- `not_contains_length_code`: 0/2 selected records had model-generated wrong programs; wrong-candidate score 0; 0 unique wrong programs. Top: none.
- `sorted_index_offset_label`: 4/24 selected records had model-generated wrong programs; wrong-candidate score 8; 1 unique wrong programs. Top: 8x `(format "SI{}" (add (sum values) offset))`.
- `sum_length_branch_label`: 0/2 selected records had model-generated wrong programs; wrong-candidate score 0; 0 unique wrong programs. Top: none.
- `sum_offset_mod_label`: 0/2 selected records had model-generated wrong programs; wrong-candidate score 0; 0 unique wrong programs. Top: none.
- `tuple_branch_label`: 0/2 selected records had model-generated wrong programs; wrong-candidate score 0; 0 unique wrong programs. Top: none.
- `tuple_sum_gate_label`: 0/2 selected records had model-generated wrong programs; wrong-candidate score 0; 0 unique wrong programs. Top: none.

## Main Results

| Condition | Data | Prompt | Samples | Greedy Hidden | Rerank Hidden |
| --- | --- | --- | ---: | ---: | ---: |
| Seed adapter | `dsl_eval_challenge.jsonl` | `trace` | 3 | 46.7% (56/120) | 51.7% (62/120) |
| Static bridge adapter | `dsl_eval_challenge.jsonl` | `trace` | 3 | 99.2% (119/120) | 98.3% (118/120) |
| Seed-mined bridge adapter | `dsl_eval_challenge.jsonl` | `trace` | 3 | 76.7% (92/120) | 84.2% (101/120) |
| Adaptive bridge adapter | `dsl_eval_challenge.jsonl` | `trace` | 3 | 85.0% (102/120) | 85.0% (102/120) |
| Adaptive bridge adapter, no trace | `dsl_eval_challenge.jsonl` | `no_trace` | 0 | 58.3% (70/120) | 58.3% (70/120) |
| Adaptive bridge adapter, shuffled trace | `dsl_eval_challenge.jsonl` | `shuffled_trace` | 0 | 17.5% (21/120) | 17.5% (21/120) |
| Seed adapter, IID | `dsl_eval_iid.jsonl` | `trace` | 0 | 100.0% (60/60) | 100.0% (60/60) |
| Static bridge adapter, IID | `dsl_eval_iid.jsonl` | `trace` | 0 | 100.0% (60/60) | 100.0% (60/60) |
| Seed-mined bridge adapter, IID | `dsl_eval_iid.jsonl` | `trace` | 0 | 100.0% (60/60) | 100.0% (60/60) |
| Adaptive bridge adapter, IID | `dsl_eval_iid.jsonl` | `trace` | 0 | 100.0% (60/60) | 100.0% (60/60) |

## Frontier By Family

| Family | Seed adapter | Static bridge adapter | Seed-mined bridge adapter | Adaptive bridge adapter |
| --- | ---: | ---: | ---: | ---: |
| `modulo_sum_label` | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| `length_contains_code` | 8.3% (1/12) | 100.0% (12/12) | 100.0% (12/12) | 83.3% (10/12) |
| `tuple_branch_label` | 100.0% (12/12) | 100.0% (12/12) | 41.7% (5/12) | 100.0% (12/12) |
| `sum_offset_mod_label` | 0.0% (0/12) | 100.0% (12/12) | 100.0% (12/12) | 83.3% (10/12) |
| `length_mod_contains_code` | 50.0% (6/12) | 91.7% (11/12) | 100.0% (12/12) | 100.0% (12/12) |
| `sum_length_branch_label` | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| `sorted_index_offset_label` | 0.0% (0/12) | 91.7% (11/12) | 100.0% (12/12) | 100.0% (12/12) |
| `contains_count_length_code` | 83.3% (10/12) | 100.0% (12/12) | 83.3% (10/12) | 83.3% (10/12) |
| `tuple_sum_gate_label` | 75.0% (9/12) | 100.0% (12/12) | 58.3% (7/12) | 100.0% (12/12) |
| `not_contains_length_code` | 0.0% (0/12) | 100.0% (12/12) | 58.3% (7/12) | 0.0% (0/12) |

## Readout

- Frontier reranked hidden all-pass: seed 51.7% (62/120), static bridge 98.3% (118/120), seed-mined bridge 84.2% (101/120), adaptive bridge 85.0% (102/120).
- Static bridge greedy hidden all-pass: 99.2% (119/120).
- Adaptive bridge greedy hidden all-pass: 85.0% (102/120).
- Adaptive bridge prompt controls: aligned trace 85.0% (102/120), no trace 58.3% (70/120), shuffled trace 17.5% (21/120).

## Failure Signatures

- `modulo_sum_label` seed greedy top: 12x `(format "M{}" (mod (sum values) modulus))`; adaptive greedy top: 12x `(format "M{}" (mod (sum values) modulus))`.
- `length_contains_code` seed greedy top: 12x `(if (and (contains text needle) (gt (count_eq text needle) threshold)) "MATCH_LONG" "MISS")`; adaptive greedy top: 9x `(if (and (contains text needle) (gt (len text) threshold)) "MATCH_LONG" "MISS")`; 2x `(if (and (gt (len text) threshold) (gt (len needle) 0)) "MATCH_LONG" "MISS")`.
- `tuple_branch_label` seed greedy top: 12x `(if (gt (tuple_get item index) threshold) high_label low_label)`; adaptive greedy top: 12x `(if (gt (tuple_get item index) threshold) high_label low_label)`.
- `sum_offset_mod_label` seed greedy top: 10x `(format "OM{}" (add (mod (sum values) modulus) offset))`; 1x `(format "OM{}" (add (sum values) offset) (mod . 0))`; adaptive greedy top: 10x `(format "OM{}" (mod (add (sum values) offset) modulus))`; 2x `(format "OM{}" (mod (sum values) offset))`.
- `length_mod_contains_code` seed greedy top: 6x `(if (and (contains text needle) (eq (mod (len text) modulus) target)) "HIT_MOD" "MISS")`; 4x `(if (and (contains text needle) (eq (mod (count_eq text needle) modulus) target)) "HIT_MOD" "MISS")`; adaptive greedy top: 11x `(if (and (contains text needle) (eq (mod (len text) modulus) target)) "HIT_MOD" "MISS")`; 1x `(if (and (gt (len text) needle) (eq (mod (len text) modulus) target)) "HIT_MOD" "MISS")`.
- `sum_length_branch_label` seed greedy top: 12x `(if (and (gt (sum values) threshold) (gt (len text) min_len)) high_label low_label)`; adaptive greedy top: 12x `(if (and (gt (sum values) threshold) (gt (len text) min_len)) high_label low_label)`.
- `sorted_index_offset_label` seed greedy top: 4x `(format "SI{}" (add (tuple_get values index) (add (sum values) offset))`; 3x `(format "SI{}" (add (tuple_get values index) offset))`; adaptive greedy top: 12x `(format "SI{}" (add (tuple_get (sort values) index) offset))`.
- `contains_count_length_code` seed greedy top: 4x `(if (and (contains tokens needle) (and (gt (count_eq tokens needle) threshold) (gt (len tokens) min_len))) "MANY_LONG" "MISS")`; 3x `(if (and (contains tokens needle) (gt (count_eq tokens needle) threshold)) "MANY_LONG" "MISS")`; adaptive greedy top: 6x `(if (and (gt (len tokens) min_len) (gt (count_eq tokens needle) threshold)) "MANY_LONG" "MISS")`; 5x `(if (and (contains tokens needle) (gt (count_eq tokens needle) threshold) (gt (len tokens) min_len)) "MANY_LONG" "MISS")`.
- `tuple_sum_gate_label` seed greedy top: 7x `(if (and (gt (tuple_get item index) threshold) (gt (sum item) sum_threshold)) high_label low_label)`; 5x `(if (and (gt (sum item) sum_threshold) (gt (get item index) threshold)) high_label low_label)`; adaptive greedy top: 12x `(if (and (gt (tuple_get item index) threshold) (gt (sum item) sum_threshold)) high_label low_label)`.
- `not_contains_length_code` seed greedy top: 12x `(if (and (contains text needle) (gt (count_eq text needle) threshold)) "ABSENT_LONG" "OTHER")`; adaptive greedy top: 10x `(if (and (contains text needle) (gt (len text) threshold)) "ABSENT_LONG" "OTHER")`; 2x `(if (and (gt (len text) threshold) (contains text needle)) "ABSENT_LONG" "OTHER")`.

## Per-Condition Details

### seed_lora_frontier

- Adapter: `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/seed_lora`.
- Data: `data/eval/dsl_eval_challenge.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 46.7% (56/120).
- Rerank hidden all-pass: 51.7% (62/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_count_length_code | 58.3% (7/12) | 83.3% (10/12) | 58.3% (7/12) | 91.7% (11/12) |
| length_contains_code | 0.0% (0/12) | 8.3% (1/12) | 8.3% (1/12) | 16.7% (2/12) |
| length_mod_contains_code | 50.0% (6/12) | 50.0% (6/12) | 66.7% (8/12) | 66.7% (8/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 0.0% (0/12) | 0.0% (0/12) | 8.3% (1/12) | 8.3% (1/12) |
| sorted_index_offset_label | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| tuple_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_sum_gate_label | 58.3% (7/12) | 75.0% (9/12) | 58.3% (7/12) | 75.0% (9/12) |

### static_bridge_lora_frontier

- Adapter: `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/static_bridge_lora`.
- Data: `data/eval/dsl_eval_challenge.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 99.2% (119/120).
- Rerank hidden all-pass: 98.3% (118/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_count_length_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| length_contains_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| length_mod_contains_code | 100.0% (12/12) | 91.7% (11/12) | 100.0% (12/12) | 100.0% (12/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sorted_index_offset_label | 91.7% (11/12) | 91.7% (11/12) | 91.7% (11/12) | 91.7% (11/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_sum_gate_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |

### seed_mined_bridge_lora_frontier

- Adapter: `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/seed_mined_bridge_lora`.
- Data: `data/eval/dsl_eval_challenge.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 76.7% (92/120).
- Rerank hidden all-pass: 84.2% (101/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_count_length_code | 58.3% (7/12) | 83.3% (10/12) | 58.3% (7/12) | 83.3% (10/12) |
| length_contains_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| length_mod_contains_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 8.3% (1/12) | 58.3% (7/12) | 8.3% (1/12) | 58.3% (7/12) |
| sorted_index_offset_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_branch_label | 41.7% (5/12) | 41.7% (5/12) | 41.7% (5/12) | 41.7% (5/12) |
| tuple_sum_gate_label | 58.3% (7/12) | 58.3% (7/12) | 75.0% (9/12) | 75.0% (9/12) |

### adaptive_bridge_lora_frontier

- Adapter: `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/adaptive_bridge_lora`.
- Data: `data/eval/dsl_eval_challenge.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 85.0% (102/120).
- Rerank hidden all-pass: 85.0% (102/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_count_length_code | 91.7% (11/12) | 83.3% (10/12) | 91.7% (11/12) | 91.7% (11/12) |
| length_contains_code | 83.3% (10/12) | 83.3% (10/12) | 91.7% (11/12) | 91.7% (11/12) |
| length_mod_contains_code | 91.7% (11/12) | 100.0% (12/12) | 91.7% (11/12) | 100.0% (12/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sorted_index_offset_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 83.3% (10/12) | 83.3% (10/12) | 83.3% (10/12) | 83.3% (10/12) |
| tuple_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_sum_gate_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |

### adaptive_bridge_lora_no_trace_frontier

- Adapter: `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/adaptive_bridge_lora`.
- Data: `data/eval/dsl_eval_challenge.jsonl`.
- Prompt mode: `no_trace`.
- Samples: 0.
- Greedy hidden all-pass: 58.3% (70/120).
- Rerank hidden all-pass: 58.3% (70/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_count_length_code | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| length_contains_code | 25.0% (3/12) | 25.0% (3/12) | 25.0% (3/12) | 25.0% (3/12) |
| length_mod_contains_code | 50.0% (6/12) | 50.0% (6/12) | 50.0% (6/12) | 50.0% (6/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sorted_index_offset_label | 25.0% (3/12) | 25.0% (3/12) | 25.0% (3/12) | 25.0% (3/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 83.3% (10/12) | 83.3% (10/12) | 83.3% (10/12) | 83.3% (10/12) |
| tuple_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_sum_gate_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |

### adaptive_bridge_lora_shuffled_trace_frontier

- Adapter: `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/adaptive_bridge_lora`.
- Data: `data/eval/dsl_eval_challenge.jsonl`.
- Prompt mode: `shuffled_trace`.
- Samples: 0.
- Greedy hidden all-pass: 17.5% (21/120).
- Rerank hidden all-pass: 17.5% (21/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_count_length_code | 16.7% (2/12) | 16.7% (2/12) | 16.7% (2/12) | 16.7% (2/12) |
| length_contains_code | 25.0% (3/12) | 25.0% (3/12) | 25.0% (3/12) | 25.0% (3/12) |
| length_mod_contains_code | 41.7% (5/12) | 41.7% (5/12) | 41.7% (5/12) | 41.7% (5/12) |
| modulo_sum_label | 8.3% (1/12) | 8.3% (1/12) | 8.3% (1/12) | 8.3% (1/12) |
| not_contains_length_code | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sorted_index_offset_label | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sum_length_branch_label | 41.7% (5/12) | 41.7% (5/12) | 41.7% (5/12) | 41.7% (5/12) |
| sum_offset_mod_label | 8.3% (1/12) | 8.3% (1/12) | 8.3% (1/12) | 8.3% (1/12) |
| tuple_branch_label | 33.3% (4/12) | 33.3% (4/12) | 33.3% (4/12) | 33.3% (4/12) |
| tuple_sum_gate_label | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |

### seed_lora_iid

- Adapter: `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/seed_lora`.
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

- Adapter: `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/static_bridge_lora`.
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

### seed_mined_bridge_lora_iid

- Adapter: `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/seed_mined_bridge_lora`.
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

### adaptive_bridge_lora_iid

- Adapter: `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/models/adaptive_bridge_lora`.
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

- Compact artifacts: `/workspace/experiments/qwen35_4b_unsaturated_frontier_active_bridge/`.
- Large artifacts: `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/`.
- Dataset manifest: `data/dataset_manifest.json`.
- Mining reports: `reports/mining/`.
- Evaluation JSON files: `reports/eval/`.
