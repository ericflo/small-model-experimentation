# Qwen 3.5 4B Counterexample-Directed DSL

## Question

Can visible traces chosen as counterexamples to plausible wrong programs improve executable DSL repair compared with random visible traces?

## Design

- Base model: `Qwen/Qwen3.5-4B`.
- Model output: one executable DSL expression.
- Training: 4-bit NF4 QLoRA adapters.
- Evaluation: parse and execute generated programs on visible and hidden cases.
- Candidate selection: choose the valid candidate with the most visible-case passes.
- Main held-out families: `modulo_sum_label`, `length_contains_code`, and `tuple_branch_label`.
- Large adapter/checkpoint files are stored outside the compact experiment directory.

## Dataset

- Random trace train records: 240.
- Counterexample trace train records: 240.
- Holdout records per trace regime: 72.
- Visible cases per record: 6.
- Hidden cases per record: 18.

## Main Results

| Condition | Prompt | Samples | Greedy Hidden | Rerank Hidden | modulo_sum_label | length_contains_code | tuple_branch_label |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random-trace adapter on random traces | `trace` | 3 | 58.3% (42/72) | 72.2% (52/72) | 95.8% (23/24) | 66.7% (16/24) | 54.2% (13/24) |
| Random-trace adapter on counterexample traces | `trace` | 3 | 51.4% (37/72) | 63.9% (46/72) | 91.7% (22/24) | 58.3% (14/24) | 41.7% (10/24) |
| Counterexample-trace adapter on counterexample traces | `trace` | 3 | 58.3% (42/72) | 61.1% (44/72) | 100.0% (24/24) | 0.0% (0/24) | 83.3% (20/24) |
| Counterexample-trace adapter, no trace | `no_trace` | 0 | 52.8% (38/72) | 52.8% (38/72) | 100.0% (24/24) | 0.0% (0/24) | 58.3% (14/24) |
| Counterexample-trace adapter, shuffled trace | `shuffled_trace` | 0 | 22.2% (16/72) | 22.2% (16/72) | 45.8% (11/24) | 0.0% (0/24) | 20.8% (5/24) |

## Readout

- Counterexample-directed training improved greedy hidden all-pass on the counterexample holdout from 51.4% (37/72) to 58.3% (42/72), but did not improve reranked hidden all-pass: 63.9% (46/72) to 61.1% (44/72).
- Coherent traces mattered for the counterexample-trained adapter: no-trace greedy hidden all-pass was 52.8% (38/72), while shuffled traces fell to 22.2% (16/72).
- The effect was family-specific. The counterexample-trained adapter reached 100.0% (24/24) on `modulo_sum_label` and 83.3% (20/24) on `tuple_branch_label`, but stayed at 0.0% (0/24) on `length_contains_code`.
- The main failure was not syntax. On `length_contains_code`, the counterexample-trained adapter generated `(if (and (contains text needle) (gt (count_eq text needle) threshold)) "MATCH_LONG" "MISS")` on 24/24 holdout records. This valid program confuses text length with needle count; sampling produced no useful diversity for reranking.
- The random-trace adapter was less collapsed on the same `length_contains_code` holdout. Its top greedy programs were: 9/24 `(if (and (contains text needle) (gt (len text) threshold)) "MATCH_LONG" "MISS")`; 9/24 `(if (contains text needle) "MATCH_LONG" (if (gt (len text) threshold) "MATCH_LONG" "MISS"))`; 3/24 `(if (contains text needle) (if (len text) gt threshold "MATCH_LONG" "MISS") "MISS")`.

## Interpretation

This experiment gives a mixed but useful answer. Counterexample-directed visible traces are strong supervision when the model has already learned the right primitive composition, as shown by the tuple-family rescue from sampled reranking. They are not sufficient by themselves to force the model to learn the correct latent primitive binding: the length family collapsed into a stable `count_eq` alias even though the selected traces were intended to distinguish plausible wrong programs.

The next iteration should make the counterexamples adaptive to the model's actual wrong program, not just to hand-authored distractors. In this run the selector distinguished the target from planned distractors, but it did not anticipate the learned `count_eq` alias. A stronger loop would sample candidate model programs during training-data construction, execute them, add traces that separate those candidates from the target, and then retrain or continue training on those model-specific counterexamples.

## Per-Condition Details

### random_lora_random_holdout

- Adapter: `/workspace/large_artifacts/qwen35_4b_counterexample_directed_dsl/models/random_trace_lora`.
- Data: `data/random/dsl_eval_holdout.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 58.3% (42/72).
- Rerank hidden all-pass: 72.2% (52/72).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| length_contains_code | 54.2% (13/24) | 66.7% (16/24) | 54.2% (13/24) | 79.2% (19/24) |
| modulo_sum_label | 87.5% (21/24) | 95.8% (23/24) | 87.5% (21/24) | 95.8% (23/24) |
| tuple_branch_label | 33.3% (8/24) | 54.2% (13/24) | 33.3% (8/24) | 54.2% (13/24) |

### random_lora_counterexample_holdout

- Adapter: `/workspace/large_artifacts/qwen35_4b_counterexample_directed_dsl/models/random_trace_lora`.
- Data: `data/counterexample/dsl_eval_holdout.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 51.4% (37/72).
- Rerank hidden all-pass: 63.9% (46/72).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| length_contains_code | 45.8% (11/24) | 58.3% (14/24) | 45.8% (11/24) | 58.3% (14/24) |
| modulo_sum_label | 75.0% (18/24) | 91.7% (22/24) | 75.0% (18/24) | 91.7% (22/24) |
| tuple_branch_label | 33.3% (8/24) | 41.7% (10/24) | 33.3% (8/24) | 41.7% (10/24) |

### counterexample_lora_counterexample_holdout

- Adapter: `/workspace/large_artifacts/qwen35_4b_counterexample_directed_dsl/models/counterexample_trace_lora`.
- Data: `data/counterexample/dsl_eval_holdout.jsonl`.
- Prompt mode: `trace`.
- Samples: 3.
- Greedy hidden all-pass: 58.3% (42/72).
- Rerank hidden all-pass: 61.1% (44/72).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| length_contains_code | 0.0% (0/24) | 0.0% (0/24) | 0.0% (0/24) | 0.0% (0/24) |
| modulo_sum_label | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) |
| tuple_branch_label | 75.0% (18/24) | 83.3% (20/24) | 79.2% (19/24) | 83.3% (20/24) |

### counterexample_lora_no_trace_holdout

- Adapter: `/workspace/large_artifacts/qwen35_4b_counterexample_directed_dsl/models/counterexample_trace_lora`.
- Data: `data/counterexample/dsl_eval_holdout.jsonl`.
- Prompt mode: `no_trace`.
- Samples: 0.
- Greedy hidden all-pass: 52.8% (38/72).
- Rerank hidden all-pass: 52.8% (38/72).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| length_contains_code | 0.0% (0/24) | 0.0% (0/24) | 0.0% (0/24) | 0.0% (0/24) |
| modulo_sum_label | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) | 100.0% (24/24) |
| tuple_branch_label | 58.3% (14/24) | 58.3% (14/24) | 58.3% (14/24) | 58.3% (14/24) |

### counterexample_lora_shuffled_trace_holdout

- Adapter: `/workspace/large_artifacts/qwen35_4b_counterexample_directed_dsl/models/counterexample_trace_lora`.
- Data: `data/counterexample/dsl_eval_holdout.jsonl`.
- Prompt mode: `shuffled_trace`.
- Samples: 0.
- Greedy hidden all-pass: 22.2% (16/72).
- Rerank hidden all-pass: 22.2% (16/72).

| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |
| --- | ---: | ---: | ---: | ---: |
| length_contains_code | 0.0% (0/24) | 0.0% (0/24) | 0.0% (0/24) | 0.0% (0/24) |
| modulo_sum_label | 45.8% (11/24) | 45.8% (11/24) | 45.8% (11/24) | 45.8% (11/24) |
| tuple_branch_label | 20.8% (5/24) | 20.8% (5/24) | 20.8% (5/24) | 20.8% (5/24) |

## Artifact Layout

- Compact artifacts: `/workspace/experiments/qwen35_4b_counterexample_directed_dsl/`.
- Large artifacts: `/workspace/large_artifacts/qwen35_4b_counterexample_directed_dsl/`.
- Dataset manifest: `data/dataset_manifest.json`.
- Evaluation JSON files: `reports/eval/`.
