# Experiment Log: Qwen3.5-4B Strategy-Token Diversity LoRA

Date: 2026-06-25

Experiment directory:
`/workspace/experiments/qwen35_4b_strategy_token_diversity_lora`

Large artifact directory:
`/workspace/large_artifacts/qwen35_4b_strategy_token_diversity_lora`

## Objective

Train a small QLoRA adapter that conditions generation on explicit strategy keys, then test whether K=32 strategy-conditioned sampling recovers more held-out MBPP base misses than hot K=32 sampling at comparable budget and whether it approaches the K=32 multi-policy union at lower cost.

## Design Commitments

- Use Qwen3.5-4B only.
- Keep this package standalone with its own config, scripts, data, logs, reports, and figures.
- Store adapters outside the experiment directory.
- Train on self-generated solutions verified by execution on MBPP train tasks.
- Evaluate on held-out MBPP tasks with hidden tests reserved for scoring.
- Use hot K=32 and K=32 multi-policy union as mandatory baselines.
- Include a shuffled-strategy-assignment control.
- Judge diversity by hidden-test coverage and functional failure-set diversity, not surface variety alone.

## Initial Package

- Created standalone experiment and large-artifact directories.
- Added package-local execution, sampling, evaluation, and model utilities.
- Added local baseline artifacts for the held-out K=4, hot K=32, and union K=32 comparisons.

## Smoke: Train Mining and SFT Row Construction

Completed `smoke_train_k8` on 8 MBPP train tasks with 8 samples/task.

- Hidden coverage: 6 / 8 = 75.0%.
- Mean hidden-pass candidates/task: 2.875.
- Estimated forward tokens: 15,412.

Built semantic and shuffled smoke SFT rows.

- Semantic rows: 11 rows from 6 tasks.
- Shuffled rows: 11 rows from the same 6 tasks.
- Smoke caught and fixed a fragile structural-classifier regex before any long run.

Decision: mining, execution verification, strategy classification, and shuffled assignment work. Proceed to main train-data mining.

## Main Train-Data Mining

Completed `main_train_k16` on 80 MBPP train tasks with 16 samples/task.

- Hidden coverage: 60 / 80 = 75.0%.
- Mean deduped candidates/task: 13.03.
- Mean hidden-pass candidates/task: 6.53.
- Mean functional diversity rate: 0.194.
- Estimated forward tokens: 303,040.

Built full SFT datasets from verified hidden-correct self-generated candidates.

- Semantic strategy rows: 244 rows from 60 tasks.
- Shuffled strategy rows: 244 rows from the same 60 tasks.
- Semantic strategy counts: COMPREHENSION 28, DIRECT 34, LOOP 45, MATH 19, RECURSION 73, SET_DICT 8, SORTING 8, STRING_REGEX 29.
- Shuffled control preserves the same targets but breaks the semantic strategy-to-target mapping.

Decision: the data is broad enough for a pilot adapter, though SET_DICT and SORTING are sparse. Train semantic and shuffled QLoRA adapters and compare them on held-out K=32 strategy sampling.

## Adapter Training

Completed semantic strategy LoRA training.

- Train rows: 244.
- Steps: 120.
- LoRA rank/alpha/dropout: 16 / 32 / 0.05.
- Final logged loss: 0.106.
- Adapter path: `/workspace/large_artifacts/qwen35_4b_strategy_token_diversity_lora/models/semantic_strategy_lora`.

Completed shuffled strategy LoRA training with the same hyperparameters and targets but shuffled strategy assignments.

- Train rows: 244.
- Steps: 120.
- Final logged loss: 0.146.
- Adapter path: `/workspace/large_artifacts/qwen35_4b_strategy_token_diversity_lora/models/shuffled_strategy_lora`.

Decision: both adapters fit the small verified dataset. Proceed to held-out K=32 strategy sampling; the comparison against shuffled assignment will decide whether semantic strategy labels add anything.

## Held-Out Strategy Sampling

Completed an all-80 semantic strategy diagnostic first.

- Standalone semantic K=32 coverage: 65 / 80 = 81.25%.
- Base-miss recovery: 11 / 24.
- Forward tokens: 627,176.
- Interpretation: this diagnostic was useful for coverage shape, but it is not the fair efficiency comparison because it spends strategy-token samples on tasks the base K=4 pool already solved.

Created the exact 24-task base-miss subset and reran the intended deployable setting: base K=4 for all tasks plus strategy K=32 only on the base-missed tasks.

Fair held-out comparison:

- `base_k4`: 56 / 80 coverage, 0 / 24 recovered, 69,645 forward tokens.
- `hot_k32`: 66 / 80 coverage, 10 / 24 recovered, 243,343 cumulative forward tokens.
- `union_k32`: 69 / 80 coverage, 13 / 24 recovered, 582,812 cumulative forward tokens.
- `base_plus_semantic_strategy_k32`: 65 / 80 coverage, 9 / 24 recovered, 274,406 cumulative forward tokens.
- `base_plus_shuffled_strategy_k32`: 66 / 80 coverage, 10 / 24 recovered, 283,391 cumulative forward tokens.

Recovered task IDs:

- Semantic subset: 22, 31, 35, 36, 42, 67, 81, 84, 87.
- Shuffled subset: 22, 31, 35, 36, 42, 67, 73, 81, 84, 87.
- Hot K=32: 15, 22, 35, 36, 42, 55, 59, 67, 70, 81.
- Union K=32: 15, 22, 34, 35, 36, 42, 55, 59, 67, 70, 81, 83, 87.

Decision: the strategy-token LoRA is a clear null for the intended claim. The semantic adapter did not beat hot sampling at matched scale, did not approach the union, and underperformed the shuffled-key control. The shuffled control matching hot K=32 means the recoveries are attributable to extra stochastic samples under longer strategy prompts, not to a meaningful learned strategy-key-to-mode mapping.

## Report

Generated the final report and figures.

- Report: `reports/final_report.md`
- Summary JSON: `reports/summary.json`
- Figures: `reports/figures/coverage_and_recovery.png`, `reports/figures/training_losses.png`

Main conclusion: small strategy-token QLoRA did not buy sampling efficiency. For this run, inference-time diverse/hot sampling remains the stronger lever.
