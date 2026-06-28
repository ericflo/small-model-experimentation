# Qwen 3.5 4B Balanced Discriminative Bridge

## Question

Can equal frontier-family coverage improve when visible traces are chosen to discriminate against harder aliases and seed-adapter mistakes, while keeping the same 240-record posttraining budget?

## Design

- Base model: `Qwen/Qwen3.5-4B`.
- Model output: one executable DSL expression.
- Training: 4-bit NF4 QLoRA adapters.
- Training budget: 240 records per trained adapter.
- Seed adapter: 240 base-family random-trace records.
- Static bridge adapter: 180 base-family records plus 60 equally allocated normal frontier bridge records.
- Alias-discriminative bridge adapter: 180 base-family records plus 60 equally allocated hard-case frontier records selected against an expanded alias bank.
- Model-discriminative bridge adapter: 180 base-family records plus 60 equally allocated hard-case frontier records selected against seed-adapter wrong programs plus the alias bank.
- Evaluation: normal frontier, harder frontier, trace controls, and IID retention.
- Candidate selection: choose the valid candidate with the most visible-case passes.
- Large adapter/checkpoint files are stored outside the compact experiment directory.

## Dataset

- Seed train records: 240.
- Bridge anchor records per bridge condition: 180.
- Bridge records per bridge condition: 60.
- Static bridge train records: 240.
- Alias-discriminative train records: 240.
- Frontier eval records: 120.
- Hard frontier eval records: 120.
- IID eval records: 60.
- Mining pool records: 240.
- Frontier families: 10.
- Visible cases per record: 6.
- Hidden cases per record: 18.

## Model-Discriminative Mining

- Allocation mode: `fixed`.
- Selector case mode: `hard`.
- Bridge allocation: `{'contains_count_length_code': 6, 'length_contains_code': 6, 'length_mod_contains_code': 6, 'modulo_sum_label': 6, 'not_contains_length_code': 6, 'sorted_index_offset_label': 6, 'sum_length_branch_label': 6, 'sum_offset_mod_label': 6, 'tuple_branch_label': 6, 'tuple_sum_gate_label': 6}`.
- `contains_count_length_code`: 6/6 selected records had seed-adapter wrong programs; wrong-candidate score 33; 1 unique model wrong programs. Top: 33x `(if (and (gt (len tokens) min_len) (contains tokens needle)) "MANY_LONG" "MISS")`.
- `length_contains_code`: 4/6 selected records had seed-adapter wrong programs; wrong-candidate score 6; 1 unique model wrong programs. Top: 6x `(if (and (contains text needle) (gt (len text) threshold)) "MATCH" "MISS")`.
- `length_mod_contains_code`: 6/6 selected records had seed-adapter wrong programs; wrong-candidate score 23; 1 unique model wrong programs. Top: 23x `(if (and (contains text needle) (gt (mod (len text) modulus) target)) "HIT_MOD" "MISS")`.
- `modulo_sum_label`: 0/6 selected records had seed-adapter wrong programs; wrong-candidate score 0; 0 unique model wrong programs. Top: none.
- `not_contains_length_code`: 6/6 selected records had seed-adapter wrong programs; wrong-candidate score 72; 6 unique model wrong programs. Top: 60x `(if (and (contains text needle) (gt (len text) threshold)) "ABSENT_LONG" "OTHER")`; 5x `(if (and (contains text needle) (gt (len text) threshold)) "OTHER" "ABSENT_LONG")`; 3x `(if (and (contains text needle) (gt (count_eq text needle) threshold)) "ABSENT_LONG" "OTHER")`.
- `sorted_index_offset_label`: 6/6 selected records had seed-adapter wrong programs; wrong-candidate score 35; 3 unique model wrong programs. Top: 17x `(format "SI{}" (add (tuple_get values index) offset))`; 15x `(format "SI{}" (sub (tuple_get values index) offset))`; 3x `(format "SI{}" (add (sum values) offset))`.
- `sum_length_branch_label`: 0/6 selected records had seed-adapter wrong programs; wrong-candidate score 0; 0 unique model wrong programs. Top: none.
- `sum_offset_mod_label`: 6/6 selected records had seed-adapter wrong programs; wrong-candidate score 66; 2 unique model wrong programs. Top: 60x `(format "OM{}" (add (sum values) (mod offset modulus)))`; 6x `(format "OM{}" (add (sum values) offset))`.
- `tuple_branch_label`: 6/6 selected records had seed-adapter wrong programs; wrong-candidate score 8; 3 unique model wrong programs. Top: 3x `(if (and (gt (tuple_get item index) threshold) (gt (sum item) threshold)) high_label low_label)`; 3x `(if (and (gt (sum item) threshold) (gt (tuple_get item index) threshold)) high_label low_label)`; 2x `(if (and (gt (sum item) threshold) (gt (tuple_get item index) 0)) high_label low_label)`.
- `tuple_sum_gate_label`: 6/6 selected records had seed-adapter wrong programs; wrong-candidate score 33; 4 unique model wrong programs. Top: 16x `(if (and (gt (sum item) sum_threshold) (gt (len item) threshold)) high_label low_label)`; 7x `(if (and (gt (sum item) sum_threshold) (gt (sum item) threshold)) high_label low_label)`; 7x `(if (and (gt (sum item) sum_threshold) (gt threshold 0)) high_label low_label)`.

## Normal Frontier Results

| Condition | Data | Prompt | Samples | Greedy Hidden | Rerank Hidden |
| --- | --- | --- | ---: | ---: | ---: |
| Seed adapter, frontier | `dsl_eval_frontier.jsonl` | `trace` | 3 | 56.7% (68/120) | 62.5% (75/120) |
| Static bridge adapter, frontier | `dsl_eval_frontier.jsonl` | `trace` | 3 | 98.3% (118/120) | 98.3% (118/120) |
| Alias-discriminative bridge adapter, frontier | `dsl_eval_frontier.jsonl` | `trace` | 3 | 88.3% (106/120) | 88.3% (106/120) |
| Model-discriminative bridge adapter, frontier | `dsl_eval_frontier.jsonl` | `trace` | 3 | 72.5% (87/120) | 78.3% (94/120) |

## Hard Frontier Results

| Condition | Data | Prompt | Samples | Greedy Hidden | Rerank Hidden |
| --- | --- | --- | ---: | ---: | ---: |
| Seed adapter, hard frontier | `dsl_eval_hard_frontier.jsonl` | `trace` | 3 | 56.7% (68/120) | 60.0% (72/120) |
| Static bridge adapter, hard frontier | `dsl_eval_hard_frontier.jsonl` | `trace` | 3 | 99.2% (119/120) | 99.2% (119/120) |
| Alias-discriminative bridge adapter, hard frontier | `dsl_eval_hard_frontier.jsonl` | `trace` | 3 | 89.2% (107/120) | 90.0% (108/120) |
| Model-discriminative bridge adapter, hard frontier | `dsl_eval_hard_frontier.jsonl` | `trace` | 3 | 79.2% (95/120) | 82.5% (99/120) |

## Trace Control Results

| Condition | Data | Prompt | Samples | Greedy Hidden | Rerank Hidden |
| --- | --- | --- | ---: | ---: | ---: |
| Static bridge adapter, no trace hard frontier | `dsl_eval_hard_frontier.jsonl` | `no_trace` | 0 | 90.8% (109/120) | 90.8% (109/120) |
| Static bridge adapter, shuffled trace hard frontier | `dsl_eval_hard_frontier.jsonl` | `shuffled_trace` | 0 | 15.8% (19/120) | 15.8% (19/120) |

## IID Retention Results

| Condition | Data | Prompt | Samples | Greedy Hidden | Rerank Hidden |
| --- | --- | --- | ---: | ---: | ---: |
| Seed adapter, IID | `dsl_eval_iid.jsonl` | `trace` | 0 | 100.0% (60/60) | 100.0% (60/60) |
| Static bridge adapter, IID | `dsl_eval_iid.jsonl` | `trace` | 0 | 100.0% (60/60) | 100.0% (60/60) |
| Alias-discriminative bridge adapter, IID | `dsl_eval_iid.jsonl` | `trace` | 0 | 100.0% (60/60) | 100.0% (60/60) |
| Model-discriminative bridge adapter, IID | `dsl_eval_iid.jsonl` | `trace` | 0 | 100.0% (60/60) | 100.0% (60/60) |

## Normal Frontier By Family

| Family | Seed adapter, frontier | Static bridge adapter, frontier | Alias-discriminative bridge adapter, frontier | Model-discriminative bridge adapter, frontier |
| --- | ---: | ---: | ---: | ---: |
| `modulo_sum_label` | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| `length_contains_code` | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 41.7% (5/12) |
| `tuple_branch_label` | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| `sum_offset_mod_label` | 16.7% (2/12) | 100.0% (12/12) | 100.0% (12/12) | 66.7% (8/12) |
| `length_mod_contains_code` | 66.7% (8/12) | 100.0% (12/12) | 100.0% (12/12) | 75.0% (9/12) |
| `sum_length_branch_label` | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| `sorted_index_offset_label` | 50.0% (6/12) | 91.7% (11/12) | 41.7% (5/12) | 75.0% (9/12) |
| `contains_count_length_code` | 33.3% (4/12) | 100.0% (12/12) | 41.7% (5/12) | 100.0% (12/12) |
| `tuple_sum_gate_label` | 58.3% (7/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| `not_contains_length_code` | 0.0% (0/12) | 91.7% (11/12) | 100.0% (12/12) | 25.0% (3/12) |

## Hard Frontier By Family

| Family | Seed adapter, hard frontier | Static bridge adapter, hard frontier | Alias-discriminative bridge adapter, hard frontier | Model-discriminative bridge adapter, hard frontier |
| --- | ---: | ---: | ---: | ---: |
| `modulo_sum_label` | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| `length_contains_code` | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 66.7% (8/12) |
| `tuple_branch_label` | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| `sum_offset_mod_label` | 16.7% (2/12) | 100.0% (12/12) | 100.0% (12/12) | 66.7% (8/12) |
| `length_mod_contains_code` | 75.0% (9/12) | 100.0% (12/12) | 100.0% (12/12) | 91.7% (11/12) |
| `sum_length_branch_label` | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| `sorted_index_offset_label` | 33.3% (4/12) | 100.0% (12/12) | 50.0% (6/12) | 83.3% (10/12) |
| `contains_count_length_code` | 8.3% (1/12) | 100.0% (12/12) | 50.0% (6/12) | 100.0% (12/12) |
| `tuple_sum_gate_label` | 66.7% (8/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| `not_contains_length_code` | 0.0% (0/12) | 91.7% (11/12) | 100.0% (12/12) | 16.7% (2/12) |

## Readout

- Hard frontier reranked hidden all-pass: seed 60.0% (72/120), static bridge 99.2% (119/120), alias-discriminative bridge 90.0% (108/120), model-discriminative bridge 82.5% (99/120).
- Hard frontier greedy hidden all-pass: seed 56.7% (68/120), static bridge 99.2% (119/120), alias-discriminative bridge 89.2% (107/120), model-discriminative bridge 79.2% (95/120).
- Static bridge trace controls on hard frontier: correct trace 99.2% (119/120), no trace 90.8% (109/120), shuffled trace 15.8% (19/120).

## Next Experiment Options

1. Recommended: run a static-normal bridge ceiling breaker. Keep `Qwen/Qwen3.5-4B`, keep equal family allocation, and replace selector hardness with harder held-out family construction: more unseen compositions, longer inputs, adversarial edge cases, and trace controls. This directly tests whether the 119/120 result is a real bridge-interface gain or an evaluation ceiling.
2. Run a bridge-budget and case-count ablation around the static recipe: 20/40/60/80 bridge records and 2/4/6/8 visible cases per record. This identifies whether the gain is coming from family coverage, trace density, or sheer bridge-token exposure.
3. Run a mild hard-case mixture instead of fully hard discriminative selection: 75% normal static records and 25% hard selector records within each family. This tests whether the regression came from hard-case distribution shift rather than discriminative selection itself.
4. Run trace-semantic regularization only after the ceiling breaker: train with a small fraction of corrupted or missing traces labeled by the correct program. This is higher risk, but the shuffled-trace collapse shows the interface is semantically sensitive enough to justify a targeted robustness experiment.

## Failure Signatures

- `modulo_sum_label` seed greedy top: 12x `(format "M{}" (mod (sum values) modulus))`; model-discriminative greedy top: 12x `(format "M{}" (mod (sum values) modulus))`.
- `length_contains_code` seed greedy top: 12x `(if (and (contains text needle) (gt (len text) threshold)) "MATCH_LONG" "MISS")`; model-discriminative greedy top: 8x `(if (and (contains text needle) (gt (len text) threshold)) "MATCH_LONG" "MISS")`; 4x `(if (and (contains text needle) (gt (count_eq text needle) threshold)) "MATCH_LONG" "MISS")`.
- `tuple_branch_label` seed greedy top: 8x `(if (gt (tuple_get item index) threshold) high_label low_label)`; 3x `(if (and (gt (sum item) threshold) (gt (tuple_get item index) 0)) high_label low_label)`; model-discriminative greedy top: 12x `(if (gt (tuple_get item index) threshold) high_label low_label)`.
- `sum_offset_mod_label` seed greedy top: 10x `(format "OM{}" (add (sum values) (mod offset modulus)))`; 1x `(format "OM{}" (add (sum values) offset))`; model-discriminative greedy top: 6x `(format "OM{}" (mod (add (sum values) offset) modulus))`; 3x `(format "OM{}" (add (mod (sum values) offset) modulus))`.
- `length_mod_contains_code` seed greedy top: 9x `(if (and (contains text needle) (eq (mod (len text) modulus) target)) "HIT_MOD" "MISS")`; 3x `(if (and (contains text needle) (gt (mod (len text) modulus) target)) "HIT_MOD" "MISS")`; model-discriminative greedy top: 10x `(if (and (contains text needle) (eq (mod (len text) modulus) target)) "HIT_MOD" "MISS")`; 2x `(if (and (contains text needle) (eq (mod (count_eq text needle) modulus) target)) "HIT_MOD" "MISS")`.
- `sum_length_branch_label` seed greedy top: 7x `(if (and (gt (sum values) threshold) (gt (len text) min_len)) high_label low_label)`; 5x `(if (and (gt (len text) min_len) (gt (sum values) threshold)) high_label low_label)`; model-discriminative greedy top: 12x `(if (and (gt (sum values) threshold) (gt (len text) min_len)) high_label low_label)`.
- `sorted_index_offset_label` seed greedy top: 7x `(format "SI{}" (add (tuple_get values index) offset))`; 4x `(format "SI{}" (add (tuple_get (sort values) index) offset))`; model-discriminative greedy top: 10x `(format "SI{}" (add (tuple_get (sort values) index) offset))`; 1x `(format "SI{}" (tuple_get (add (sort values) offset) index))`.
- `contains_count_length_code` seed greedy top: 7x `(if (and (gt (len tokens) min_len) (contains tokens needle)) "MANY_LONG" "MISS")`; 5x `(if (and (contains tokens needle) (and (gt (count_eq tokens needle) threshold) (gt (len tokens) min_len)) "MANY_LONG" "MISS")`; model-discriminative greedy top: 12x `(if (and (contains tokens needle) (gt (count_eq tokens needle) threshold) (gt (len tokens) min_len)) "MANY_LONG" "MISS")`.
- `tuple_sum_gate_label` seed greedy top: 7x `(if (and (gt (tuple_get item index) threshold) (gt (sum item) sum_threshold)) high_label low_label)`; 2x `(if (and (gt (sum item) sum_threshold) (gt (len item) threshold)) high_label low_label)`; model-discriminative greedy top: 12x `(if (and (gt (tuple_get item index) threshold) (gt (sum item) sum_threshold)) high_label low_label)`.
- `not_contains_length_code` seed greedy top: 11x `(if (and (contains text needle) (gt (len text) threshold)) "ABSENT_LONG" "OTHER")`; 1x `(if (and (contains text needle) (gt (len text) threshold)) "FOUND" "ABSENT_LONG")`; model-discriminative greedy top: 8x `(if (and (contains text needle) (gt (len text) threshold)) "ABSENT_LONG" "OTHER")`; 3x `(if (and (contains text needle) (gt (count_eq text needle) threshold)) "ABSENT_LONG" "OTHER")`.

## Per-Condition Details

### seed_lora_frontier

- Adapter: `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/seed_lora`.
- Data: `data/eval/dsl_eval_frontier.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 56.7% (68/120).
- Rerank hidden all-pass: 62.5% (75/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_count_length_code | 16.7% (2/12) | 33.3% (4/12) | 41.7% (5/12) | 58.3% (7/12) |
| length_contains_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| length_mod_contains_code | 58.3% (7/12) | 66.7% (8/12) | 58.3% (7/12) | 66.7% (8/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sorted_index_offset_label | 25.0% (3/12) | 50.0% (6/12) | 25.0% (3/12) | 50.0% (6/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 8.3% (1/12) | 16.7% (2/12) | 8.3% (1/12) | 16.7% (2/12) |
| tuple_branch_label | 91.7% (11/12) | 100.0% (12/12) | 91.7% (11/12) | 100.0% (12/12) |
| tuple_sum_gate_label | 66.7% (8/12) | 58.3% (7/12) | 75.0% (9/12) | 75.0% (9/12) |

### static_bridge_lora_frontier

- Adapter: `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/static_bridge_lora`.
- Data: `data/eval/dsl_eval_frontier.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 98.3% (118/120).
- Rerank hidden all-pass: 98.3% (118/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_count_length_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| length_contains_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| length_mod_contains_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 91.7% (11/12) | 91.7% (11/12) | 91.7% (11/12) | 91.7% (11/12) |
| sorted_index_offset_label | 91.7% (11/12) | 91.7% (11/12) | 91.7% (11/12) | 91.7% (11/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_sum_gate_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |

### alias_discriminative_bridge_lora_frontier

- Adapter: `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/alias_discriminative_bridge_lora`.
- Data: `data/eval/dsl_eval_frontier.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 88.3% (106/120).
- Rerank hidden all-pass: 88.3% (106/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_count_length_code | 41.7% (5/12) | 41.7% (5/12) | 66.7% (8/12) | 66.7% (8/12) |
| length_contains_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| length_mod_contains_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sorted_index_offset_label | 41.7% (5/12) | 41.7% (5/12) | 41.7% (5/12) | 41.7% (5/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_sum_gate_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |

### model_discriminative_bridge_lora_frontier

- Adapter: `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/model_discriminative_bridge_lora`.
- Data: `data/eval/dsl_eval_frontier.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 72.5% (87/120).
- Rerank hidden all-pass: 78.3% (94/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_count_length_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| length_contains_code | 33.3% (4/12) | 41.7% (5/12) | 33.3% (4/12) | 41.7% (5/12) |
| length_mod_contains_code | 58.3% (7/12) | 75.0% (9/12) | 58.3% (7/12) | 75.0% (9/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 8.3% (1/12) | 25.0% (3/12) | 25.0% (3/12) | 41.7% (5/12) |
| sorted_index_offset_label | 66.7% (8/12) | 75.0% (9/12) | 66.7% (8/12) | 75.0% (9/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 58.3% (7/12) | 66.7% (8/12) | 58.3% (7/12) | 66.7% (8/12) |
| tuple_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_sum_gate_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |

### seed_lora_hard_frontier

- Adapter: `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/seed_lora`.
- Data: `data/eval/dsl_eval_hard_frontier.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 56.7% (68/120).
- Rerank hidden all-pass: 60.0% (72/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_count_length_code | 0.0% (0/12) | 8.3% (1/12) | 0.0% (0/12) | 8.3% (1/12) |
| length_contains_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| length_mod_contains_code | 75.0% (9/12) | 75.0% (9/12) | 75.0% (9/12) | 75.0% (9/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sorted_index_offset_label | 33.3% (4/12) | 33.3% (4/12) | 33.3% (4/12) | 33.3% (4/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 8.3% (1/12) | 16.7% (2/12) | 8.3% (1/12) | 16.7% (2/12) |
| tuple_branch_label | 75.0% (9/12) | 100.0% (12/12) | 75.0% (9/12) | 100.0% (12/12) |
| tuple_sum_gate_label | 75.0% (9/12) | 66.7% (8/12) | 91.7% (11/12) | 91.7% (11/12) |

### static_bridge_lora_hard_frontier

- Adapter: `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/static_bridge_lora`.
- Data: `data/eval/dsl_eval_hard_frontier.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 99.2% (119/120).
- Rerank hidden all-pass: 99.2% (119/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_count_length_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| length_contains_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| length_mod_contains_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 91.7% (11/12) | 91.7% (11/12) | 91.7% (11/12) | 91.7% (11/12) |
| sorted_index_offset_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_sum_gate_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |

### alias_discriminative_bridge_lora_hard_frontier

- Adapter: `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/alias_discriminative_bridge_lora`.
- Data: `data/eval/dsl_eval_hard_frontier.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 89.2% (107/120).
- Rerank hidden all-pass: 90.0% (108/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_count_length_code | 50.0% (6/12) | 50.0% (6/12) | 58.3% (7/12) | 58.3% (7/12) |
| length_contains_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| length_mod_contains_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 91.7% (11/12) | 100.0% (12/12) | 91.7% (11/12) | 100.0% (12/12) |
| sorted_index_offset_label | 50.0% (6/12) | 50.0% (6/12) | 50.0% (6/12) | 50.0% (6/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_sum_gate_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |

### model_discriminative_bridge_lora_hard_frontier

- Adapter: `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/model_discriminative_bridge_lora`.
- Data: `data/eval/dsl_eval_hard_frontier.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 79.2% (95/120).
- Rerank hidden all-pass: 82.5% (99/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_count_length_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| length_contains_code | 66.7% (8/12) | 66.7% (8/12) | 66.7% (8/12) | 66.7% (8/12) |
| length_mod_contains_code | 83.3% (10/12) | 91.7% (11/12) | 83.3% (10/12) | 91.7% (11/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 8.3% (1/12) | 16.7% (2/12) | 25.0% (3/12) | 33.3% (4/12) |
| sorted_index_offset_label | 83.3% (10/12) | 83.3% (10/12) | 83.3% (10/12) | 83.3% (10/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 50.0% (6/12) | 66.7% (8/12) | 50.0% (6/12) | 66.7% (8/12) |
| tuple_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_sum_gate_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |

### static_bridge_lora_no_trace_hard_frontier

- Adapter: `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/static_bridge_lora`.
- Data: `data/eval/dsl_eval_hard_frontier.jsonl`.
- Prompt mode: `no_trace`.
- Samples: 0.
- Greedy hidden all-pass: 90.8% (109/120).
- Rerank hidden all-pass: 90.8% (109/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_count_length_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| length_contains_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| length_mod_contains_code | 58.3% (7/12) | 58.3% (7/12) | 58.3% (7/12) | 58.3% (7/12) |
| modulo_sum_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| not_contains_length_code | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sorted_index_offset_label | 50.0% (6/12) | 50.0% (6/12) | 50.0% (6/12) | 50.0% (6/12) |
| sum_length_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_offset_mod_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_branch_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| tuple_sum_gate_label | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |

### static_bridge_lora_shuffled_trace_hard_frontier

- Adapter: `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/static_bridge_lora`.
- Data: `data/eval/dsl_eval_hard_frontier.jsonl`.
- Prompt mode: `shuffled_trace`.
- Samples: 0.
- Greedy hidden all-pass: 15.8% (19/120).
- Rerank hidden all-pass: 15.8% (19/120).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| contains_count_length_code | 16.7% (2/12) | 16.7% (2/12) | 16.7% (2/12) | 16.7% (2/12) |
| length_contains_code | 33.3% (4/12) | 33.3% (4/12) | 41.7% (5/12) | 41.7% (5/12) |
| length_mod_contains_code | 25.0% (3/12) | 25.0% (3/12) | 25.0% (3/12) | 25.0% (3/12) |
| modulo_sum_label | 8.3% (1/12) | 8.3% (1/12) | 8.3% (1/12) | 8.3% (1/12) |
| not_contains_length_code | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sorted_index_offset_label | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sum_length_branch_label | 25.0% (3/12) | 25.0% (3/12) | 25.0% (3/12) | 25.0% (3/12) |
| sum_offset_mod_label | 16.7% (2/12) | 16.7% (2/12) | 16.7% (2/12) | 16.7% (2/12) |
| tuple_branch_label | 33.3% (4/12) | 33.3% (4/12) | 33.3% (4/12) | 33.3% (4/12) |
| tuple_sum_gate_label | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |

### seed_lora_iid

- Adapter: `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/seed_lora`.
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

- Adapter: `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/static_bridge_lora`.
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

### alias_discriminative_bridge_lora_iid

- Adapter: `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/alias_discriminative_bridge_lora`.
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

### model_discriminative_bridge_lora_iid

- Adapter: `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/model_discriminative_bridge_lora`.
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

- Compact artifacts: `/workspace/experiments/qwen35_4b_balanced_discriminative_bridge/`.
- Large artifacts: `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/`.
- Dataset manifest: `data/dataset_manifest.json`.
- Mining reports: `reports/mining/`.
- Evaluation JSON files: `reports/eval/`.
