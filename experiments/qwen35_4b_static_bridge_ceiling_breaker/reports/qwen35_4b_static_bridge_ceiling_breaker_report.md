# Qwen 3.5 4B Static Bridge Ceiling Breaker

## Question

Can fixed-budget static bridge posttraining learn a trace-conditioned repair interface that transfers from support bridge families to deeper held-out composition families?

## Design

- Base model: `Qwen/Qwen3.5-4B`.
- Model output: one executable DSL expression.
- Training: 4-bit NF4 QLoRA adapters.
- Candidate selection: choose the valid candidate with the most visible-case passes.
- Large adapter/checkpoint files are stored outside the compact experiment directory.
- Seed adapter: 240 base-family random-trace records.
- Static 60 adapter: 180 base-family records plus 60 equal support bridge records.
- Static 80 adapter: 160 base-family records plus 80 equal support bridge records.
- Main test: held-out ceiling families absent from bridge training.

## Dataset

- Seed train records: 240.
- Static 60 train records: 240 (180 base + 60 bridge).
- Static 80 train records: 240 (160 base + 80 bridge).
- IID eval records: 60.
- Support eval records: 120.
- Ceiling eval records: 120.
- Visible cases per record: 6.
- Hidden cases per record: 18.
- Support bridge families: 10.
- Ceiling families: 10.
- Static 60 selector summary: `{'avg_eliminated_wrong_programs': 4.1, 'avg_remaining_wrong_programs': 0.0, 'records': 60}`.
- Static 80 selector summary: `{'avg_eliminated_wrong_programs': 4.1, 'avg_remaining_wrong_programs': 0.0, 'records': 80}`.

## Support Split Results

| Condition | Data | Prompt | Samples | Greedy Hidden | Rerank Hidden |
| --- | --- | --- | ---: | ---: | ---: |
| Seed, support | `dsl_eval_support.jsonl` | `trace` | 3 | 50.8% (61/120) | 53.3% (64/120) |
| Static 60, support | `dsl_eval_support.jsonl` | `trace` | 3 | 99.2% (119/120) | 100.0% (120/120) |
| Static 80, support | `dsl_eval_support.jsonl` | `trace` | 3 | 100.0% (120/120) | 100.0% (120/120) |

## Ceiling Split Results

| Condition | Data | Prompt | Samples | Greedy Hidden | Rerank Hidden |
| --- | --- | --- | ---: | ---: | ---: |
| Seed, ceiling | `dsl_eval_ceiling.jsonl` | `trace` | 3 | 12.5% (15/120) | 20.0% (24/120) |
| Static 60, ceiling | `dsl_eval_ceiling.jsonl` | `trace` | 3 | 38.3% (46/120) | 44.2% (53/120) |
| Static 80, ceiling | `dsl_eval_ceiling.jsonl` | `trace` | 3 | 40.0% (48/120) | 40.8% (49/120) |

## Ceiling Trace Controls

| Condition | Data | Prompt | Samples | Greedy Hidden | Rerank Hidden |
| --- | --- | --- | ---: | ---: | ---: |
| Static 60, no trace ceiling | `dsl_eval_ceiling.jsonl` | `no_trace` | 0 | 15.0% (18/120) | 15.0% (18/120) |
| Static 60, shuffled trace ceiling | `dsl_eval_ceiling.jsonl` | `shuffled_trace` | 0 | 6.7% (8/120) | 6.7% (8/120) |

## IID Retention Results

| Condition | Data | Prompt | Samples | Greedy Hidden | Rerank Hidden |
| --- | --- | --- | ---: | ---: | ---: |
| Seed, IID | `dsl_eval_iid.jsonl` | `trace` | 0 | 100.0% (60/60) | 100.0% (60/60) |
| Static 60, IID | `dsl_eval_iid.jsonl` | `trace` | 0 | 100.0% (60/60) | 100.0% (60/60) |
| Static 80, IID | `dsl_eval_iid.jsonl` | `trace` | 0 | 100.0% (60/60) | 100.0% (60/60) |

## Ceiling By Family

| Family | Seed, ceiling | Static 60, ceiling | Static 80, ceiling |
| --- | ---: | ---: | ---: |
| `sum_length_mod_gate_label` | 0.0% (0/12) | 8.3% (1/12) | 0.0% (0/12) |
| `sorted_index_sum_branch_label` | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| `token_absent_length_code` | 0.0% (0/12) | 91.7% (11/12) | 58.3% (7/12) |
| `token_count_mod_length_code` | 41.7% (5/12) | 83.3% (10/12) | 58.3% (7/12) |
| `text_value_gate_label` | 83.3% (10/12) | 100.0% (12/12) | 100.0% (12/12) |
| `tuple_value_mod_label` | 50.0% (6/12) | 8.3% (1/12) | 50.0% (6/12) |
| `sorted_join_contains_code` | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| `text_absent_mod_code` | 0.0% (0/12) | 100.0% (12/12) | 100.0% (12/12) |
| `sum_len_mod_label` | 0.0% (0/12) | 8.3% (1/12) | 0.0% (0/12) |
| `tuple_sum_mod_gate_label` | 25.0% (3/12) | 41.7% (5/12) | 41.7% (5/12) |

## Support By Family

| Family | Seed, support | Static 60, support | Static 80, support |
| --- | ---: | ---: | ---: |
| `modulo_sum_label` | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| `length_contains_code` | 66.7% (8/12) | 100.0% (12/12) | 100.0% (12/12) |
| `tuple_branch_label` | 66.7% (8/12) | 100.0% (12/12) | 100.0% (12/12) |
| `sum_offset_mod_label` | 0.0% (0/12) | 100.0% (12/12) | 100.0% (12/12) |
| `length_mod_contains_code` | 66.7% (8/12) | 100.0% (12/12) | 100.0% (12/12) |
| `sum_length_branch_label` | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| `sorted_index_offset_label` | 0.0% (0/12) | 100.0% (12/12) | 100.0% (12/12) |
| `contains_count_length_code` | 58.3% (7/12) | 100.0% (12/12) | 100.0% (12/12) |
| `tuple_sum_gate_label` | 75.0% (9/12) | 100.0% (12/12) | 100.0% (12/12) |
| `not_contains_length_code` | 0.0% (0/12) | 100.0% (12/12) | 100.0% (12/12) |

## Readout

- Ceiling reranked hidden all-pass: seed 20.0% (24/120), static 60 44.2% (53/120), static 80 40.8% (49/120).
- Ceiling greedy hidden all-pass: seed 12.5% (15/120), static 60 38.3% (46/120), static 80 40.0% (48/120).
- Static 60, ceiling trace controls: aligned 38.3% (46/120), no trace 15.0% (18/120), shuffled trace 6.7% (8/120).

## Figures

- `figures/support_ceiling_rerank_hidden.png`
- `figures/ceiling_trace_controls.png`
- `figures/ceiling_by_family.png`

## Failure Signatures

- `sum_length_mod_gate_label` seed greedy top: 6x `(if (and (gt (sum values) 0) (eq (mod (len text) modulus) target)) high_label low_label)`; 6x `(if (and (gt (sum values) 0) (eq (mod (sum values) modulus) target)) high_label low_label)`; static 60 greedy top: 6x `(if (and (contains text "a") (eq (mod (sum values) modulus) target)) high_label low_label)`; 3x `(if (and (contains text "e") (eq (mod (sum values) modulus) target)) high_label low_label)`; static 80 greedy top: 5x `(if (and (gt (sum values) 0) (eq (mod (len text) modulus) target)) high_label low_label)`; 2x `(if (and (contains text "a") (gt (sum values) 0)) high_label low_label)`.
- `sorted_index_sum_branch_label` seed greedy top: 3x `(if (and (gt (tuple_get values index) threshold) (gt (sum values) 0)) high_label low_label)`; 2x `(if (and (gt (sum values) threshold) (eq index 0)) high_label low_label)`; static 60 greedy top: 7x `(if (gt (tuple_get (sort values) index) threshold) high_label low_label)`; 5x `(if (gt (tuple_get values index) threshold) high_label low_label)`; static 80 greedy top: 7x `(if (gt (tuple_get (sort values) index) threshold) high_label low_label)`; 4x `(if (gt (tuple_get values index) threshold) high_label low_label)`.
- `token_absent_length_code` seed greedy top: 3x `(if (and (contains tokens needle) (gt (count_eq tokens needle) min_len)) "FOUND" "OTHER")`; 3x `(if (and (contains tokens needle) (gt (len tokens) min_len)) "ABSENT_LONG" "OTHER")`; static 60 greedy top: 9x `(if (and (not (contains tokens needle)) (gt (len tokens) min_len)) "ABSENT_LONG" "OTHER")`; 2x `(if (and (contains tokens needle) (gt (len tokens) min_len)) "ABSENT_LONG" "OTHER")`; static 80 greedy top: 6x `(if (and (not (contains tokens needle)) (gt (len tokens) min_len)) "ABSENT_LONG" "OTHER")`; 5x `(if (and (not (contains tokens needle)) (gt (count_eq tokens needle) min_len)) "ABSENT_LONG" "OTHER")`.
- `token_count_mod_length_code` seed greedy top: 5x `(if (and (contains tokens needle) (gt (len tokens) min_len) (and (eq (mod (count_eq tokens needle) modulus) target))) "COUNT_MOD_LONG" "MISS")`; 2x `(if (and (contains tokens needle) (gt (len tokens) min_len) (and (gt (mod (len tokens) modulus) 0) (gt target 0))) "COUNT_MOD_LONG" "MISS")`; static 60 greedy top: 5x `(if (and (contains tokens needle) (eq (mod (count_eq tokens needle) modulus) target) (gt (len tokens) min_len)) "COUNT_MOD_LONG" "MISS")`; 4x `(if (and (contains tokens needle) (gt (len tokens) min_len) (eq (mod (count_eq tokens needle) modulus) target)) "COUNT_MOD_LONG" "MISS")`; static 80 greedy top: 6x `(if (and (contains tokens needle) (gt (len tokens) min_len) (eq (mod (count_eq tokens needle) modulus) target)) "COUNT_MOD_LONG" "MISS")`; 6x `(if (and (contains tokens needle) (gt (len tokens) min_len) (eq (mod (len tokens) modulus) target)) "COUNT_MOD_LONG" "MISS")`.
- `text_value_gate_label` seed greedy top: 10x `(if (and (contains text needle) (gt (sum values) threshold) (gt (len text) min_len)) high_label low_label)`; 1x `(if (and (gt (len text) min_len) (contains text needle)) (gt (sum values) threshold) high_label low_label)`; static 60 greedy top: 8x `(if (and (contains text needle) (gt (sum values) threshold) (gt (len text) min_len)) high_label low_label)`; 4x `(if (and (contains text needle) (gt (len text) min_len) (gt (sum values) threshold)) high_label low_label)`; static 80 greedy top: 12x `(if (and (contains text needle) (gt (sum values) threshold) (gt (len text) min_len)) high_label low_label)`.
- `tuple_value_mod_label` seed greedy top: 4x `(format "TV{}" (add (tuple_get item index) (mod (sum values) modulus)))`; 2x `(format "TV{}" (tuple_get item index))`; static 60 greedy top: 9x `(format "TV{}" (mod (add (tuple_get item index) (sum values)) modulus))`; 2x `(format "TV{}" (mod (add (tuple_get item index) (first values)) modulus))`; static 80 greedy top: 6x `(format "TV{}" (add (tuple_get item index) (mod (sum values) modulus)))`; 4x `(format "TV{}" (mod (add (tuple_get item index) (sum values)) modulus))`.
- `sorted_join_contains_code` seed greedy top: 4x `(if (and tokens (contains tokens needle)) "JOIN_HAS" "JOIN_MISS")`; 2x `(if (contains (join "" tokens) needle) "JOIN_HAS" "JOIN_MISS")`; static 60 greedy top: 6x `(if (contains (join "" tokens) needle) "JOIN_HAS" "JOIN_MISS")`; 5x `(if (contains (sort tokens) needle) "JOIN_HAS" "JOIN_MISS")`; static 80 greedy top: 5x `(if (and (contains tokens needle) (count_eq tokens needle)) "JOIN_HAS" "JOIN_MISS")`; 3x `(if (and (contains (join "" tokens) needle) (eq (count_eq tokens needle) 0)) "JOIN_HAS" "JOIN_MISS")`.
- `text_absent_mod_code` seed greedy top: 5x `(if (and (contains text needle) (eq (mod (len text) modulus) target)) "FOUND" "OTHER")`; 3x `(if (and (contains text needle) (eq (mod (len text) modulus) target)) "FOUND" "ABSENT_MOD")`; static 60 greedy top: 12x `(if (and (not (contains text needle)) (eq (mod (len text) modulus) target)) "ABSENT_MOD" "OTHER")`; static 80 greedy top: 12x `(if (and (not (contains text needle)) (eq (mod (len text) modulus) target)) "ABSENT_MOD" "OTHER")`.
- `sum_len_mod_label` seed greedy top: 6x `(format "SL{}" (mod (sum values) modulus))`; 5x `(format "SL{}" (add (sum values) (mod (len text) modulus)))`; static 60 greedy top: 7x `(format "SL{}" (mod (sum values) modulus))`; 4x `(format "SL{}" (mod (len text) (mod (sum values) modulus)))`; static 80 greedy top: 12x `(format "SL{}" (mod (sum values) modulus))`.
- `tuple_sum_mod_gate_label` seed greedy top: 5x `(if (and (gt (get item index) threshold) (eq (mod (sum item) modulus) target)) high_label low_label)`; 5x `(if (and (gt (tuple_get item index) threshold) (eq (mod (tuple_get item index) modulus) target)) high_label low_label)`; static 60 greedy top: 8x `(if (and (gt (tuple_get item index) threshold) (eq (mod (tuple_get item index) modulus) target)) high_label low_label)`; 4x `(if (and (gt (tuple_get item index) threshold) (eq (mod (sum item) modulus) target)) high_label low_label)`; static 80 greedy top: 7x `(if (and (gt (tuple_get item index) threshold) (eq (mod (tuple_get item index) modulus) target)) high_label low_label)`; 5x `(if (and (gt (tuple_get item index) threshold) (eq (mod (sum item) modulus) target)) high_label low_label)`.

## Per-Condition Details

### seed_lora_ceiling

- Adapter: `/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/models/seed_lora`.
- Data: `data/eval/dsl_eval_ceiling.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 12.5% (15/120).
- Rerank hidden all-pass: 20.0% (24/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| sorted_index_sum_branch_label | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sorted_join_contains_code | 0.0% (0/12) | 0.0% (0/12) | 16.7% (2/12) | 16.7% (2/12) |
| sum_len_mod_label | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sum_length_mod_gate_label | 0.0% (0/12) | 0.0% (0/12) | 16.7% (2/12) | 16.7% (2/12) |
| text_absent_mod_code | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| text_value_gate_label | 83.3% (10/12) | 83.3% (10/12) | 91.7% (11/12) | 100.0% (12/12) |
| token_absent_length_code | 0.0% (0/12) | 0.0% (0/12) | 16.7% (2/12) | 16.7% (2/12) |
| token_count_mod_length_code | 8.3% (1/12) | 41.7% (5/12) | 0.0% (0/12) | 41.7% (5/12) |
| tuple_sum_mod_gate_label | 0.0% (0/12) | 25.0% (3/12) | 8.3% (1/12) | 33.3% (4/12) |
| tuple_value_mod_label | 33.3% (4/12) | 50.0% (6/12) | 33.3% (4/12) | 50.0% (6/12) |

### seed_lora_iid

- Adapter: `/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/models/seed_lora`.
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

### seed_lora_support

- Adapter: `/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/models/seed_lora`.
- Data: `data/eval/dsl_eval_support.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 50.8% (61/120).
- Rerank hidden all-pass: 53.3% (64/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_count_length_code | 41.7% (5/12) | 58.3% (7/12) | 58.3% (7/12) | 75.0% (9/12) |
| length_contains_code | 66.7% (8/12) | 66.7% (8/12) | 75.0% (9/12) | 75.0% (9/12) |
| length_mod_contains_code | 66.7% (8/12) | 66.7% (8/12) | 75.0% (9/12) | 75.0% (9/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 0.0% (0/12) | 0.0% (0/12) | 16.7% (2/12) | 16.7% (2/12) |
| sorted_index_offset_label | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| tuple_branch_label | 66.7% (8/12) | 66.7% (8/12) | 66.7% (8/12) | 66.7% (8/12) |
| tuple_sum_gate_label | 66.7% (8/12) | 75.0% (9/12) | 66.7% (8/12) | 75.0% (9/12) |

### static60_lora_ceiling

- Adapter: `/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/models/static60_lora`.
- Data: `data/eval/dsl_eval_ceiling.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 38.3% (46/120).
- Rerank hidden all-pass: 44.2% (53/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| sorted_index_sum_branch_label | 0.0% (0/12) | 0.0% (0/12) | 8.3% (1/12) | 8.3% (1/12) |
| sorted_join_contains_code | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sum_len_mod_label | 0.0% (0/12) | 8.3% (1/12) | 0.0% (0/12) | 8.3% (1/12) |
| sum_length_mod_gate_label | 0.0% (0/12) | 8.3% (1/12) | 16.7% (2/12) | 25.0% (3/12) |
| text_absent_mod_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| text_value_gate_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| token_absent_length_code | 75.0% (9/12) | 91.7% (11/12) | 83.3% (10/12) | 100.0% (12/12) |
| token_count_mod_length_code | 75.0% (9/12) | 83.3% (10/12) | 75.0% (9/12) | 91.7% (11/12) |
| tuple_sum_mod_gate_label | 33.3% (4/12) | 41.7% (5/12) | 41.7% (5/12) | 50.0% (6/12) |
| tuple_value_mod_label | 0.0% (0/12) | 8.3% (1/12) | 0.0% (0/12) | 8.3% (1/12) |

### static60_lora_iid

- Adapter: `/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/models/static60_lora`.
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

### static60_lora_no_trace_ceiling

- Adapter: `/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/models/static60_lora`.
- Data: `data/eval/dsl_eval_ceiling.jsonl`.
- Prompt mode: `no_trace`.
- Samples: 0.
- Greedy hidden all-pass: 15.0% (18/120).
- Rerank hidden all-pass: 15.0% (18/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| sorted_index_sum_branch_label | 0.0% (0/12) | 0.0% (0/12) | 8.3% (1/12) | 8.3% (1/12) |
| sorted_join_contains_code | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sum_len_mod_label | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sum_length_mod_gate_label | 0.0% (0/12) | 0.0% (0/12) | 16.7% (2/12) | 16.7% (2/12) |
| text_absent_mod_code | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| text_value_gate_label | 41.7% (5/12) | 41.7% (5/12) | 66.7% (8/12) | 66.7% (8/12) |
| token_absent_length_code | 0.0% (0/12) | 0.0% (0/12) | 8.3% (1/12) | 8.3% (1/12) |
| token_count_mod_length_code | 58.3% (7/12) | 58.3% (7/12) | 58.3% (7/12) | 58.3% (7/12) |
| tuple_sum_mod_gate_label | 0.0% (0/12) | 0.0% (0/12) | 8.3% (1/12) | 8.3% (1/12) |
| tuple_value_mod_label | 50.0% (6/12) | 50.0% (6/12) | 50.0% (6/12) | 50.0% (6/12) |

### static60_lora_shuffled_trace_ceiling

- Adapter: `/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/models/static60_lora`.
- Data: `data/eval/dsl_eval_ceiling.jsonl`.
- Prompt mode: `shuffled_trace`.
- Samples: 0.
- Greedy hidden all-pass: 6.7% (8/120).
- Rerank hidden all-pass: 6.7% (8/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| sorted_index_sum_branch_label | 0.0% (0/12) | 0.0% (0/12) | 8.3% (1/12) | 8.3% (1/12) |
| sorted_join_contains_code | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sum_len_mod_label | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sum_length_mod_gate_label | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| text_absent_mod_code | 16.7% (2/12) | 16.7% (2/12) | 16.7% (2/12) | 16.7% (2/12) |
| text_value_gate_label | 25.0% (3/12) | 25.0% (3/12) | 25.0% (3/12) | 25.0% (3/12) |
| token_absent_length_code | 8.3% (1/12) | 8.3% (1/12) | 8.3% (1/12) | 8.3% (1/12) |
| token_count_mod_length_code | 16.7% (2/12) | 16.7% (2/12) | 16.7% (2/12) | 16.7% (2/12) |
| tuple_sum_mod_gate_label | 0.0% (0/12) | 0.0% (0/12) | 8.3% (1/12) | 8.3% (1/12) |
| tuple_value_mod_label | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |

### static60_lora_support

- Adapter: `/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/models/static60_lora`.
- Data: `data/eval/dsl_eval_support.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 99.2% (119/120).
- Rerank hidden all-pass: 100.0% (120/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_count_length_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| length_contains_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| length_mod_contains_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 91.7% (11/12) | 100.0% (12/12) | 91.7% (11/12) | 100.0% (12/12) |
| sorted_index_offset_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_sum_gate_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |

### static80_lora_ceiling

- Adapter: `/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/models/static80_lora`.
- Data: `data/eval/dsl_eval_ceiling.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 40.0% (48/120).
- Rerank hidden all-pass: 40.8% (49/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| sorted_index_sum_branch_label | 0.0% (0/12) | 0.0% (0/12) | 8.3% (1/12) | 8.3% (1/12) |
| sorted_join_contains_code | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 8.3% (1/12) |
| sum_len_mod_label | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sum_length_mod_gate_label | 0.0% (0/12) | 0.0% (0/12) | 8.3% (1/12) | 25.0% (3/12) |
| text_absent_mod_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| text_value_gate_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| token_absent_length_code | 50.0% (6/12) | 58.3% (7/12) | 66.7% (8/12) | 75.0% (9/12) |
| token_count_mod_length_code | 58.3% (7/12) | 58.3% (7/12) | 58.3% (7/12) | 58.3% (7/12) |
| tuple_sum_mod_gate_label | 41.7% (5/12) | 41.7% (5/12) | 50.0% (6/12) | 50.0% (6/12) |
| tuple_value_mod_label | 50.0% (6/12) | 50.0% (6/12) | 50.0% (6/12) | 50.0% (6/12) |

### static80_lora_iid

- Adapter: `/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/models/static80_lora`.
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

### static80_lora_support

- Adapter: `/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/models/static80_lora`.
- Data: `data/eval/dsl_eval_support.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 100.0% (120/120).
- Rerank hidden all-pass: 100.0% (120/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_count_length_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| length_contains_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| length_mod_contains_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sorted_index_offset_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_sum_gate_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
