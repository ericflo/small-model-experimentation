# Qwen 3.5 4B GraphIR Self Repair

## Question

Can a Qwen 3.5 4B adapter improve held-out executable repair by configuring a typed register graph, then applying a verifier-guided repair step?

## Design

- Base model: `Qwen/Qwen3.5-4B`.
- Training: 4-bit NF4 QLoRA adapters.
- Fixed budget: 240 records per adapter.
- DSL baseline: emits one prefix DSL expression.
- GraphIR construct adapter: emits typed register assignments ending in `out`.
- GraphIR repair adapter: receives a candidate graph plus visible execution mismatches and emits a corrected graph.
- Inference policy: generate configured construction candidates, execute visible cases, keep the best graph, optionally repair it, and score hidden cases.
- Large adapter/checkpoint files are stored outside the compact experiment directory.

## Dataset

- Base train records: 180.
- Support bridge train records: 60.
- Train records per adapter: 240.
- IID eval records: 60.
- Support eval records: 120.
- Ceiling eval records: 120.
- Visible cases per record: 6.
- Hidden cases per record: 18.
- Support bridge families: 10.
- Ceiling families: 10.

## Ceiling Results

| Condition | Data | Prompt | Samples | Main Hidden | Secondary Hidden |
| --- | --- | --- | ---: | ---: | ---: |
| DSL baseline, ceiling | `dsl_ceiling.jsonl` | `trace` | 1 | 29.2% (35/120) | 27.5% (33/120) |
| GraphIR construct, ceiling | `graph_ceiling.jsonl` | `trace` | 1 | 21.7% (26/120) | 21.7% (26/120) |
| GraphIR construct+repair, ceiling | `graph_ceiling.jsonl` | `trace` | 1 | 24.2% (29/120) | 21.7% (26/120) |

## Support Results

| Condition | Data | Prompt | Samples | Main Hidden | Secondary Hidden |
| --- | --- | --- | ---: | ---: | ---: |
| DSL baseline, support | `dsl_support.jsonl` | `trace` | 3 | 100.0% (120/120) | 100.0% (120/120) |
| GraphIR construct, support | `graph_support.jsonl` | `trace` | 0 | 98.3% (118/120) | 98.3% (118/120) |
| GraphIR construct+repair, support | `graph_support.jsonl` | `trace` | 1 | 100.0% (120/120) | 98.3% (118/120) |

## IID Results

| Condition | Data | Prompt | Samples | Main Hidden | Secondary Hidden |
| --- | --- | --- | ---: | ---: | ---: |
| DSL baseline, IID | `dsl_iid.jsonl` | `trace` | 0 | 100.0% (60/60) | 100.0% (60/60) |
| GraphIR construct, IID | `graph_iid.jsonl` | `trace` | 0 | 100.0% (60/60) | 100.0% (60/60) |
| GraphIR construct+repair, IID | `graph_iid.jsonl` | `trace` | 0 | 100.0% (60/60) | 100.0% (60/60) |

## Repair Diagnostic

| Condition | Data | Prompt | Samples | Main Hidden | Secondary Hidden |
| --- | --- | --- | ---: | ---: | ---: |
| GraphIR repair, corrupted ceiling | `graph_repair_ceiling_corrupt.jsonl` | `trace` | 0 | 26.7% (32/120) | 3.3% (4/120) |

## Ceiling By Family

| Family | DSL baseline, ceiling | GraphIR construct, ceiling | GraphIR construct+repair, ceiling |
| --- | ---: | ---: | ---: |
| `sum_length_mod_gate_label` | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| `sorted_index_sum_branch_label` | 0.0% (0/12) | 0.0% (0/12) | 8.3% (1/12) |
| `token_absent_length_code` | 0.0% (0/12) | 0.0% (0/12) | 8.3% (1/12) |
| `token_count_mod_length_code` | 0.0% (0/12) | 83.3% (10/12) | 83.3% (10/12) |
| `text_value_gate_label` | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| `tuple_value_mod_label` | 25.0% (3/12) | 0.0% (0/12) | 0.0% (0/12) |
| `sorted_join_contains_code` | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| `text_absent_mod_code` | 100.0% (12/12) | 0.0% (0/12) | 0.0% (0/12) |
| `sum_len_mod_label` | 41.7% (5/12) | 33.3% (4/12) | 33.3% (4/12) |
| `tuple_sum_mod_gate_label` | 25.0% (3/12) | 0.0% (0/12) | 8.3% (1/12) |

## Readout

- Ceiling hidden all-pass: DSL baseline 29.2% (35/120), GraphIR construction 21.7% (26/120), GraphIR construction plus repair 24.2% (29/120).
- GraphIR construction greedy ceiling hidden all-pass: 21.7% (26/120).
- GraphIR pipeline construction-only selected ceiling hidden all-pass: 21.7% (26/120).
- The GraphIR repair stage improved the actual ceiling pipeline from 21.7% (26/120) to 24.2% (29/120), but did not beat the DSL baseline.
- On synthetic corrupted ceiling GraphIR candidates, repair improved hidden all-pass from 3.3% (4/120) to 26.7% (32/120), indicating repair skill exists but does not transfer enough to actual construction errors.

## Figures

- `figures/ceiling_hidden_success.png`
- `figures/ceiling_by_family.png`

## Failure Signatures

- `sum_length_mod_gate_label` DSL top: 6x `(if (and (gt (sum values) target) (eq (mod (len text) modulus) target)) high_label low_label)`; 3x `(if (and (gt (sum values) target) (eq (mod (sum values) modulus) target)) high_label low_label)`; GraphIR construct top: 4x `r0 = SUM values
r1 = LEN text
r2 = MOD r1 modulus
r3 = EQ r2 target
r4 = AND r3 GT r0 0
r5 = IF r4 high_label low_label
out = r5`; GraphIR pipeline top: 4x `r0 = SUM values
r1 = LEN text
r2 = MOD r1 modulus
r3 = EQ r2 target
r4 = AND r3 GT r0 0
r5 = IF r4 high_label low_label
out = r5`.
- `sorted_index_sum_branch_label` DSL top: 6x `(if (gt (tuple_get (sort values) index) threshold) high_label low_label)`; 6x `(if (gt (tuple_get values index) threshold) high_label low_label)`; GraphIR construct top: 12x `r0 = GET values index
r1 = GT r0 threshold
r2 = IF r1 high_label low_label
out = r2`; GraphIR pipeline top: 11x `r0 = GET values index
r1 = GT r0 threshold
r2 = IF r1 high_label low_label
out = r2`.
- `token_absent_length_code` DSL top: 9x `(if (and (not (contains tokens needle)) (gt (count_eq tokens needle) min_len)) "ABSENT_LONG" "OTHER")`; 2x `(if (and (contains tokens needle) (gt (count_eq tokens needle) min_len)) "ABSENT_LONG" "OTHER")`; GraphIR construct top: 11x `r0 = CONTAINS tokens needle
r1 = LEN tokens
r2 = GT r1 min_len
r3 = AND r0 r2
r4 = IF r3 "ABSENT_LONG" "OTHER"
out = r4`; GraphIR pipeline top: 11x `r0 = CONTAINS tokens needle
r1 = LEN tokens
r2 = GT r1 min_len
r3 = AND r0 r2
r4 = IF r3 "ABSENT_LONG" "OTHER"
out = r4`.
- `token_count_mod_length_code` DSL top: 7x `(if (and (contains tokens needle) (gt (count_eq tokens needle) min_len) (eq (mod (count_eq tokens needle) modulus) target)) "COUNT_MOD_LONG" "MISS")`; 3x `(if (and (contains tokens needle) (gt (count_eq tokens needle) target) (gt (mod (count_eq tokens needle) modulus) 0) (gt (len tokens) min_len)) "COUNT_MOD_LONG" "MISS")`; GraphIR construct top: 9x `r0 = CONTAINS tokens needle
r1 = COUNT_EQ tokens needle
r2 = MOD r1 modulus
r3 = EQ r2 target
r4 = LEN tokens
r5 = GT r4 min_len
r6 = AND r0 r3 r5
r7 = IF r6 "COUNT_MOD_LONG" "MISS"
out = r7`; GraphIR pipeline top: 9x `r0 = CONTAINS tokens needle
r1 = COUNT_EQ tokens needle
r2 = MOD r1 modulus
r3 = EQ r2 target
r4 = LEN tokens
r5 = GT r4 min_len
r6 = AND r0 r3 r5
r7 = IF r6 "COUNT_MOD_LONG" "MISS"
out = r7`.
- `text_value_gate_label` DSL top: 12x `(if (and (contains text needle) (gt (sum values) threshold) (gt (len text) min_len)) high_label low_label)`; GraphIR construct top: 12x `r0 = CONTAINS text needle
r1 = LEN text
r2 = GT r1 min_len
r3 = SUM values
r4 = GT r3 threshold
r5 = AND r0 r2 r4
r6 = IF r5 high_label low_label
out = r6`; GraphIR pipeline top: 12x `r0 = CONTAINS text needle
r1 = LEN text
r2 = GT r1 min_len
r3 = SUM values
r4 = GT r3 threshold
r5 = AND r0 r2 r4
r6 = IF r5 high_label low_label
out = r6`.
- `tuple_value_mod_label` DSL top: 8x `(format "TV{}" (mod (add (tuple_get item index) (sum values)) modulus))`; 3x `(format "TV{}" (add (tuple_get item index) (mod (sum values) modulus)))`; GraphIR construct top: 6x `r0 = GET item index
r1 = MOD r0 modulus
r2 = FORMAT "TV{}" r1
out = r2`; GraphIR pipeline top: 5x `r0 = GET item index
r1 = MOD r0 modulus
r2 = FORMAT "TV{}" r1
out = r2`.
- `sorted_join_contains_code` DSL top: 12x `(if (contains tokens needle) "JOIN_HAS" "JOIN_MISS")`; GraphIR construct top: 8x `r0 = CONTAINS tokens needle
r1 = IF r0 "JOIN_HAS" "JOIN_MISS"
out = r1`; GraphIR pipeline top: 8x `r0 = CONTAINS tokens needle
r1 = IF r0 "JOIN_HAS" "JOIN_MISS"
out = r1`.
- `text_absent_mod_code` DSL top: 12x `(if (and (not (contains text needle)) (eq (mod (len text) modulus) target)) "ABSENT_MOD" "OTHER")`; GraphIR construct top: 12x `r0 = CONTAINS text needle
r1 = LEN text
r2 = MOD r1 modulus
r3 = EQ r2 target
r4 = AND r0 r3
r5 = IF r4 "ABSENT_MOD" "OTHER"
out = r5`; GraphIR pipeline top: 12x `r0 = CONTAINS text needle
r1 = LEN text
r2 = MOD r1 modulus
r3 = EQ r2 target
r4 = AND r0 r3
r5 = IF r4 "ABSENT_MOD" "OTHER"
out = r5`.
- `sum_len_mod_label` DSL top: 7x `(format "SL{}" (mod (sum values) modulus))`; 5x `(format "SL{}" (mod (add (sum values) (len text)) modulus))`; GraphIR construct top: 6x `r0 = SUM values
r1 = LEN text
r2 = MOD r1 modulus
r3 = FORMAT "SL{}" r2
out = r3`; GraphIR pipeline top: 5x `r0 = SUM values
r1 = LEN text
r2 = MOD r1 modulus
r3 = FORMAT "SL{}" r2
out = r3`.
- `tuple_sum_mod_gate_label` DSL top: 7x `(if (and (gt (tuple_get item index) threshold) (eq (mod (tuple_get item index) modulus) target)) high_label low_label)`; 3x `(if (and (gt (tuple_get item index) threshold) (eq (mod (sum item) modulus) target)) high_label low_label)`; GraphIR construct top: 9x `r0 = GET item index
r1 = GT r0 threshold
r2 = MOD r0 modulus
r3 = EQ r2 target
r4 = AND r1 r3
r5 = IF r4 high_label low_label
out = r5`; GraphIR pipeline top: 8x `r0 = GET item index
r1 = GT r0 threshold
r2 = MOD r0 modulus
r3 = EQ r2 target
r4 = AND r1 r3
r5 = IF r4 high_label low_label
out = r5`.

## Per-Condition Details

### dsl_lora_ceiling

- Data: `data/eval/dsl_ceiling.jsonl`.
- Prompt mode: `trace`.
- Samples: 1.

| Family | Main Hidden | Main Visible |
| --- | ---: | ---: |
| sorted_index_sum_branch_label | 0.0% (0/12) | 0.0% (0/12) |
| sorted_join_contains_code | 0.0% (0/12) | 0.0% (0/12) |
| sum_len_mod_label | 41.7% (5/12) | 41.7% (5/12) |
| sum_length_mod_gate_label | 0.0% (0/12) | 8.3% (1/12) |
| text_absent_mod_code | 100.0% (12/12) | 100.0% (12/12) |
| text_value_gate_label | 100.0% (12/12) | 100.0% (12/12) |
| token_absent_length_code | 0.0% (0/12) | 8.3% (1/12) |
| token_count_mod_length_code | 0.0% (0/12) | 16.7% (2/12) |
| tuple_sum_mod_gate_label | 25.0% (3/12) | 50.0% (6/12) |
| tuple_value_mod_label | 25.0% (3/12) | 25.0% (3/12) |

### dsl_lora_iid

- Data: `data/eval/dsl_iid.jsonl`.
- Prompt mode: `trace`.
- Samples: 0.

| Family | Main Hidden | Main Visible |
| --- | ---: | ---: |
| contains_and_count_code | 100.0% (4/4) | 100.0% (4/4) |
| contains_code | 100.0% (4/4) | 100.0% (4/4) |
| contains_count_label | 100.0% (4/4) | 100.0% (4/4) |
| length_and_mod_code | 100.0% (4/4) | 100.0% (4/4) |
| length_label | 100.0% (4/4) | 100.0% (4/4) |
| length_mod_label | 100.0% (4/4) | 100.0% (4/4) |
| mod_scalar_label | 100.0% (4/4) | 100.0% (4/4) |
| scalar_branch_label | 100.0% (4/4) | 100.0% (4/4) |
| sorted_first_label | 100.0% (4/4) | 100.0% (4/4) |
| sum_add_label | 100.0% (4/4) | 100.0% (4/4) |
| sum_and_scalar_code | 100.0% (4/4) | 100.0% (4/4) |
| sum_label | 100.0% (4/4) | 100.0% (4/4) |
| sum_threshold_label | 100.0% (4/4) | 100.0% (4/4) |
| tuple_get_label | 100.0% (4/4) | 100.0% (4/4) |
| tuple_sum_label | 100.0% (4/4) | 100.0% (4/4) |

### dsl_lora_support

- Data: `data/eval/dsl_support.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.

| Family | Main Hidden | Main Visible |
| --- | ---: | ---: |
| contains_count_length_code | 100.0% (12/12) | 100.0% (12/12) |
| length_contains_code | 100.0% (12/12) | 100.0% (12/12) |
| length_mod_contains_code | 100.0% (12/12) | 100.0% (12/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 100.0% (12/12) | 100.0% (12/12) |
| sorted_index_offset_label | 100.0% (12/12) | 100.0% (12/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 100.0% (12/12) | 100.0% (12/12) |
| tuple_branch_label | 100.0% (12/12) | 100.0% (12/12) |
| tuple_sum_gate_label | 100.0% (12/12) | 100.0% (12/12) |

### graphir_construct_ceiling

- Data: `data/eval/graph_ceiling.jsonl`.
- Prompt mode: `trace`.
- Samples: 1.

| Family | Main Hidden | Main Visible |
| --- | ---: | ---: |
| sorted_index_sum_branch_label | 0.0% (0/12) | 0.0% (0/12) |
| sorted_join_contains_code | 0.0% (0/12) | 0.0% (0/12) |
| sum_len_mod_label | 33.3% (4/12) | 33.3% (4/12) |
| sum_length_mod_gate_label | 0.0% (0/12) | 25.0% (3/12) |
| text_absent_mod_code | 0.0% (0/12) | 0.0% (0/12) |
| text_value_gate_label | 100.0% (12/12) | 100.0% (12/12) |
| token_absent_length_code | 0.0% (0/12) | 8.3% (1/12) |
| token_count_mod_length_code | 83.3% (10/12) | 100.0% (12/12) |
| tuple_sum_mod_gate_label | 0.0% (0/12) | 8.3% (1/12) |
| tuple_value_mod_label | 0.0% (0/12) | 0.0% (0/12) |

### graphir_construct_iid

- Data: `data/eval/graph_iid.jsonl`.
- Prompt mode: `trace`.
- Samples: 0.

| Family | Main Hidden | Main Visible |
| --- | ---: | ---: |
| contains_and_count_code | 100.0% (4/4) | 100.0% (4/4) |
| contains_code | 100.0% (4/4) | 100.0% (4/4) |
| contains_count_label | 100.0% (4/4) | 100.0% (4/4) |
| length_and_mod_code | 100.0% (4/4) | 100.0% (4/4) |
| length_label | 100.0% (4/4) | 100.0% (4/4) |
| length_mod_label | 100.0% (4/4) | 100.0% (4/4) |
| mod_scalar_label | 100.0% (4/4) | 100.0% (4/4) |
| scalar_branch_label | 100.0% (4/4) | 100.0% (4/4) |
| sorted_first_label | 100.0% (4/4) | 100.0% (4/4) |
| sum_add_label | 100.0% (4/4) | 100.0% (4/4) |
| sum_and_scalar_code | 100.0% (4/4) | 100.0% (4/4) |
| sum_label | 100.0% (4/4) | 100.0% (4/4) |
| sum_threshold_label | 100.0% (4/4) | 100.0% (4/4) |
| tuple_get_label | 100.0% (4/4) | 100.0% (4/4) |
| tuple_sum_label | 100.0% (4/4) | 100.0% (4/4) |

### graphir_construct_support

- Data: `data/eval/graph_support.jsonl`.
- Prompt mode: `trace`.
- Samples: 0.

| Family | Main Hidden | Main Visible |
| --- | ---: | ---: |
| contains_count_length_code | 100.0% (12/12) | 100.0% (12/12) |
| length_contains_code | 100.0% (12/12) | 100.0% (12/12) |
| length_mod_contains_code | 83.3% (10/12) | 83.3% (10/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 100.0% (12/12) | 100.0% (12/12) |
| sorted_index_offset_label | 100.0% (12/12) | 100.0% (12/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 100.0% (12/12) | 100.0% (12/12) |
| tuple_branch_label | 100.0% (12/12) | 100.0% (12/12) |
| tuple_sum_gate_label | 100.0% (12/12) | 100.0% (12/12) |

### graphir_pipeline_ceiling

- Data: `data/eval/graph_ceiling.jsonl`.
- Prompt mode: `trace`.
- Samples: 1.

| Family | Main Hidden | Main Visible |
| --- | ---: | ---: |
| sorted_index_sum_branch_label | 8.3% (1/12) | 8.3% (1/12) |
| sorted_join_contains_code | 0.0% (0/12) | 0.0% (0/12) |
| sum_len_mod_label | 33.3% (4/12) | 33.3% (4/12) |
| sum_length_mod_gate_label | 0.0% (0/12) | 25.0% (3/12) |
| text_absent_mod_code | 0.0% (0/12) | 0.0% (0/12) |
| text_value_gate_label | 100.0% (12/12) | 100.0% (12/12) |
| token_absent_length_code | 8.3% (1/12) | 16.7% (2/12) |
| token_count_mod_length_code | 83.3% (10/12) | 100.0% (12/12) |
| tuple_sum_mod_gate_label | 8.3% (1/12) | 33.3% (4/12) |
| tuple_value_mod_label | 0.0% (0/12) | 0.0% (0/12) |

### graphir_pipeline_iid

- Data: `data/eval/graph_iid.jsonl`.
- Prompt mode: `trace`.
- Samples: 0.

| Family | Main Hidden | Main Visible |
| --- | ---: | ---: |
| contains_and_count_code | 100.0% (4/4) | 100.0% (4/4) |
| contains_code | 100.0% (4/4) | 100.0% (4/4) |
| contains_count_label | 100.0% (4/4) | 100.0% (4/4) |
| length_and_mod_code | 100.0% (4/4) | 100.0% (4/4) |
| length_label | 100.0% (4/4) | 100.0% (4/4) |
| length_mod_label | 100.0% (4/4) | 100.0% (4/4) |
| mod_scalar_label | 100.0% (4/4) | 100.0% (4/4) |
| scalar_branch_label | 100.0% (4/4) | 100.0% (4/4) |
| sorted_first_label | 100.0% (4/4) | 100.0% (4/4) |
| sum_add_label | 100.0% (4/4) | 100.0% (4/4) |
| sum_and_scalar_code | 100.0% (4/4) | 100.0% (4/4) |
| sum_label | 100.0% (4/4) | 100.0% (4/4) |
| sum_threshold_label | 100.0% (4/4) | 100.0% (4/4) |
| tuple_get_label | 100.0% (4/4) | 100.0% (4/4) |
| tuple_sum_label | 100.0% (4/4) | 100.0% (4/4) |

### graphir_pipeline_support

- Data: `data/eval/graph_support.jsonl`.
- Prompt mode: `trace`.
- Samples: 1.

| Family | Main Hidden | Main Visible |
| --- | ---: | ---: |
| contains_count_length_code | 100.0% (12/12) | 100.0% (12/12) |
| length_contains_code | 100.0% (12/12) | 100.0% (12/12) |
| length_mod_contains_code | 100.0% (12/12) | 100.0% (12/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 100.0% (12/12) | 100.0% (12/12) |
| sorted_index_offset_label | 100.0% (12/12) | 100.0% (12/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 100.0% (12/12) | 100.0% (12/12) |
| tuple_branch_label | 100.0% (12/12) | 100.0% (12/12) |
| tuple_sum_gate_label | 100.0% (12/12) | 100.0% (12/12) |

### graphir_repair_corrupt_ceiling

- Data: `data/eval/graph_repair_ceiling_corrupt.jsonl`.
- Prompt mode: `trace`.
- Samples: 0.

| Family | Main Hidden | Main Visible |
| --- | ---: | ---: |
| sorted_index_sum_branch_label | 33.3% (4/12) | 66.7% (8/12) |
| sorted_join_contains_code | 0.0% (0/12) | 16.7% (2/12) |
| sum_len_mod_label | 0.0% (0/12) | 0.0% (0/12) |
| sum_length_mod_gate_label | 0.0% (0/12) | 16.7% (2/12) |
| text_absent_mod_code | 25.0% (3/12) | 25.0% (3/12) |
| text_value_gate_label | 83.3% (10/12) | 100.0% (12/12) |
| token_absent_length_code | 33.3% (4/12) | 41.7% (5/12) |
| token_count_mod_length_code | 50.0% (6/12) | 75.0% (9/12) |
| tuple_sum_mod_gate_label | 33.3% (4/12) | 50.0% (6/12) |
| tuple_value_mod_label | 8.3% (1/12) | 8.3% (1/12) |
