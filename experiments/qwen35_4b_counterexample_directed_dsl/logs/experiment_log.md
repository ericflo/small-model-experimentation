# Experiment Log

## 2026-06-22

- Created standalone experiment directory.
- Selected `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Designed the data generator to emit two trace regimes:
  - random visible cases,
  - counterexample-directed visible cases selected to distinguish the target program from plausible wrong programs.

## Dataset Build

- Generated standalone datasets with `python scripts/build_dataset.py`.
- Random trace regime:
  - 240 train records,
  - 60 IID eval records,
  - 72 compositional holdout records.
- Counterexample trace regime:
  - 240 train records,
  - 60 IID eval records,
  - 72 compositional holdout records.
- Each record has 6 visible cases and 18 hidden cases.
- Heldout families: `modulo_sum_label`, `length_contains_code`, `tuple_branch_label`.
- The counterexample selector used a 160-case candidate pool.

## Training

- Ran a smoke adapter first on 8 random-trace records to validate QLoRA loading, training, saving, and evaluation.
- Trained a random-trace LoRA adapter on `data/random/dsl_train.jsonl`.
  - Final IID eval loss: 0.0005662.
  - Training runtime: about 875 seconds.
  - Output directory: `/workspace/large_artifacts/qwen35_4b_counterexample_directed_dsl/models/random_trace_lora`.
- Trained a counterexample-trace LoRA adapter on `data/counterexample/dsl_train.jsonl`.
  - Final IID eval loss: 0.00005084.
  - Training runtime: about 840 seconds.
  - Output directory: `/workspace/large_artifacts/qwen35_4b_counterexample_directed_dsl/models/counterexample_trace_lora`.

## Evaluation

- Random-trace adapter on random-trace holdout:
  - Greedy hidden all-pass: 42/72.
  - Reranked hidden all-pass: 52/72.
- Random-trace adapter on counterexample-trace holdout:
  - Greedy hidden all-pass: 37/72.
  - Reranked hidden all-pass: 46/72.
- Counterexample-trace adapter on counterexample-trace holdout:
  - Greedy hidden all-pass: 42/72.
  - Reranked hidden all-pass: 44/72.
  - `modulo_sum_label`: 24/24 reranked hidden all-pass.
  - `length_contains_code`: 0/24 reranked hidden all-pass.
  - `tuple_branch_label`: 20/24 reranked hidden all-pass.
- Counterexample-trace adapter with no visible trace:
  - Greedy hidden all-pass: 38/72.
- Counterexample-trace adapter with shuffled visible traces:
  - Greedy hidden all-pass: 16/72.

## Iteration Notes

- Coherent traces help the counterexample-trained adapter: removing traces drops the aggregate from 42/72 greedy hidden all-pass to 38/72, and shuffling traces drops it to 16/72.
- The main negative result is the `length_contains_code` collapse. The counterexample-trained adapter emitted `(if (and (contains text needle) (gt (count_eq text needle) threshold)) "MATCH_LONG" "MISS")` on all 24 length-family holdout records.
- The length-family failure is semantic, not syntactic: the generated program is valid but binds the threshold comparison to needle count instead of text length.
- Sampling did not recover the target on that family because all sampled candidates repeated the same wrong program.
- The next experiment should make counterexamples adaptive to model-generated wrong programs instead of only to hand-authored distractors.
