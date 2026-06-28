# Qwen 3.5 4B Executable Program Posttraining

## Question

Can a Qwen 3.5 4B adapter trained to emit executable DSL repair programs produce programs that generalize to held-out composition families, and does visible-test reranking improve hidden-case success?

## Design

- Base model: `Qwen/Qwen3.5-4B`.
- Model output: one executable DSL expression.
- Evaluator: parses and executes generated programs on visible and hidden cases.
- Reranking: samples candidate programs and selects the valid candidate with the most visible-case passes.
- Main held-out families: `modulo_sum_label`, `length_contains_code`, and `tuple_branch_label`.
- Adapter weights and checkpoints are stored outside the compact directory under `/workspace/large_artifacts/qwen35_4b_executable_program_posttraining/models/`.

## Dataset

- Train records: 240.
- IID eval records: 60.
- Holdout eval records: 72.
- Visible cases per record: 6.
- Hidden cases per record: 18.

## Iteration Readout

The first trace-trained executable-program adapter transferred cleanly on two held-out families but failed the length+contains family completely. Inspection showed that failed generations repeatedly substituted `count_eq text needle` for the needed `len text` predicate inside a conjunction. A second adapter was trained from scratch with three training-only conjunction families added under the same 240-record budget.

Key held-out results:

| Condition | Prompt | Samples | Greedy Hidden | Rerank Hidden | modulo_sum_label | length_contains_code | tuple_branch_label |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial trace adapter | `trace` | 12 | 65.3% (47/72) | 66.7% (48/72) | 100.0% (24/24) | 0.0% (0/24) | 100.0% (24/24) |
| Conjunction-support trace adapter | `trace` | 3 | 72.2% (52/72) | 75.0% (54/72) | 100.0% (24/24) | 29.2% (7/24) | 95.8% (23/24) |
| Conjunction-support adapter | `no_trace` | 0 | 33.3% (24/72) | 33.3% (24/72) | 100.0% (24/24) | 0.0% (0/24) | 0.0% (0/24) |
| Conjunction-support adapter | `shuffled_trace` | 0 | 37.5% (27/72) | 37.5% (27/72) | 70.8% (17/24) | 20.8% (5/24) | 20.8% (5/24) |

Readout:

- Executable DSL posttraining produced a large held-out signal on `modulo_sum_label` and `tuple_branch_label`.
- The initial failure on `length_contains_code` was not random formatting noise; it was a specific mechanism error.
- Adding non-held-out conjunction training families moved `length_contains_code` from 0/24 to 7/24 under visible reranking, while preserving 24/24 on modulo and 23/24 on tuple.
- Aligned visible traces mattered: the conjunction-support adapter scored 54/72 with aligned trace plus 3 samples, 24/72 with no trace greedy, and 27/72 with shuffled trace greedy.
- A full 12-sample evaluation of the second adapter was started but stopped after two records because generations were taking over 90 seconds per record. The reported second-adapter rerank condition uses 3 samples and a 64-token cap.

## Results

### trace_and_bridge_lora_no_trace_holdout_greedy

- Adapter: `/workspace/large_artifacts/qwen35_4b_executable_program_posttraining/models/dsl_trace_and_bridge_lora`.
- Prompt mode: `no_trace`.
- Records: 72.
- Greedy hidden all-pass: 33.3% (24/72).
- Visible-rerank hidden all-pass: 33.3% (24/72).
- Greedy visible all-pass: 34.7% (25/72).
- Rerank visible all-pass: 34.7% (25/72).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| length_contains_code | 0.0% (0/24) | 0.0% (0/24) | 4.2% (1/24) | 4.2% (1/24) |
| modulo_sum_label | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) |
| tuple_branch_label | 0.0% (0/24) | 0.0% (0/24) | 0.0% (0/24) | 0.0% (0/24) |

### trace_and_bridge_lora_shuffled_trace_holdout_greedy

- Adapter: `/workspace/large_artifacts/qwen35_4b_executable_program_posttraining/models/dsl_trace_and_bridge_lora`.
- Prompt mode: `shuffled_trace`.
- Records: 72.
- Greedy hidden all-pass: 37.5% (27/72).
- Visible-rerank hidden all-pass: 37.5% (27/72).
- Greedy visible all-pass: 37.5% (27/72).
- Rerank visible all-pass: 37.5% (27/72).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| length_contains_code | 20.8% (5/24) | 20.8% (5/24) | 20.8% (5/24) | 20.8% (5/24) |
| modulo_sum_label | 70.8% (17/24) | 70.8% (17/24) | 70.8% (17/24) | 70.8% (17/24) |
| tuple_branch_label | 20.8% (5/24) | 20.8% (5/24) | 20.8% (5/24) | 20.8% (5/24) |

### trace_and_bridge_lora_trace_holdout_samples3

- Adapter: `/workspace/large_artifacts/qwen35_4b_executable_program_posttraining/models/dsl_trace_and_bridge_lora`.
- Prompt mode: `trace`.
- Records: 72.
- Greedy hidden all-pass: 72.2% (52/72).
- Visible-rerank hidden all-pass: 75.0% (54/72).
- Greedy visible all-pass: 72.2% (52/72).
- Rerank visible all-pass: 75.0% (54/72).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| length_contains_code | 20.8% (5/24) | 29.2% (7/24) | 20.8% (5/24) | 29.2% (7/24) |
| modulo_sum_label | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) |
| tuple_branch_label | 95.8% (23/24) | 95.8% (23/24) | 95.8% (23/24) | 95.8% (23/24) |

### trace_lora_trace_holdout

- Adapter: `/workspace/large_artifacts/qwen35_4b_executable_program_posttraining/models/dsl_trace_lora`.
- Prompt mode: `trace`.
- Records: 72.
- Greedy hidden all-pass: 65.3% (47/72).
- Visible-rerank hidden all-pass: 66.7% (48/72).
- Greedy visible all-pass: 66.7% (48/72).
- Rerank visible all-pass: 68.1% (49/72).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| length_contains_code | 0.0% (0/24) | 0.0% (0/24) | 4.2% (1/24) | 4.2% (1/24) |
| modulo_sum_label | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) |
| tuple_branch_label | 95.8% (23/24) | 100.0% (24/24) | 95.8% (23/24) | 100.0% (24/24) |

## Artifact Layout

- Compact artifacts: `/workspace/experiments/qwen35_4b_executable_program_posttraining/`.
- Large artifacts: `/workspace/large_artifacts/qwen35_4b_executable_program_posttraining/`.
- Dataset manifest: `data/dataset_manifest.json`.
- Evaluation JSON files: `reports/eval/`.
