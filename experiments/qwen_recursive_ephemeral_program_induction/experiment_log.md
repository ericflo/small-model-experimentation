# Experiment Log

## Setup

- Created a fresh standalone experiment directory.
- Large artifacts are kept under `/workspace/large_artifacts/qwen_recursive_ephemeral_program_induction`.
- Primary metric: strict full-task exact on held-out rows.
- Secondary metrics: row exact, train-pass rate, generated-program validity, and oracle headroom among train-passing generated programs.


## Run `smoke_no_qwen`

- Time UTC: `2026-06-27T10:43:22.094485+00:00`
- Elapsed seconds: `0.6`
- Config: `{"answer_max_new_tokens": 64, "batch_max_new_tokens": 320, "code_max_new_tokens": 520, "heldout_cap": 3, "min_heldout": 3, "no_qwen": true, "repair_max_new_tokens": 560, "repair_rounds": 1, "repair_variants": "minimal,broaden,conditional", "run_name": "smoke_no_qwen", "run_shuffled": true, "sample_seed": 20260627, "task_limit": 3, "train_n": 4, "variants": "monolithic,helpers,robust"}`
- Tasks: `3`
- Direct row-by-row full-task exact: `0.0%`
- Recursive selected-program full-task exact: `0.0%`
- Recursive train-pass rate: `0.0%`
- Recursive oracle among train-passing candidates: `0.0%`

## Run `smoke_qwen_3`

- Time UTC: `2026-06-27T10:47:05.311065+00:00`
- Elapsed seconds: `198.9`
- Config: `{"answer_max_new_tokens": 48, "batch_max_new_tokens": 220, "code_max_new_tokens": 360, "heldout_cap": 3, "min_heldout": 3, "no_qwen": false, "repair_max_new_tokens": 380, "repair_rounds": 1, "repair_variants": "minimal,broaden,conditional", "run_name": "smoke_qwen_3", "run_shuffled": true, "sample_seed": 20260627, "task_limit": 3, "train_n": 4, "variants": "monolithic,helpers,robust"}`
- Tasks: `3`
- Direct row-by-row full-task exact: `66.7%`
- Recursive selected-program full-task exact: `33.3%`
- Recursive train-pass rate: `66.7%`
- Recursive oracle among train-passing candidates: `33.3%`

## Run `smoke_qwen_5_v2`

- Time UTC: `2026-06-27T10:54:41.415308+00:00`
- Elapsed seconds: `402.9`
- Config: `{"answer_max_new_tokens": 48, "batch_max_new_tokens": 220, "code_max_new_tokens": 360, "heldout_cap": 3, "min_heldout": 3, "no_qwen": false, "repair_max_new_tokens": 380, "repair_rounds": 1, "repair_variants": "minimal,broaden,conditional", "run_name": "smoke_qwen_5_v2", "run_shuffled": true, "sample_seed": 20260627, "task_limit": 5, "train_n": 4, "variants": "monolithic,helpers,robust"}`
- Tasks: `5`
- Direct row-by-row full-task exact: `60.0%`
- Recursive selected-program full-task exact: `20.0%`
- Recursive train-pass rate: `40.0%`
- Recursive oracle among train-passing candidates: `20.0%`

## Run `smoke_qwen_5_helpers`

- Time UTC: `2026-06-27T11:05:15.279262+00:00`
- Elapsed seconds: `530.7`
- Config: `{"answer_max_new_tokens": 48, "batch_max_new_tokens": 220, "code_max_new_tokens": 420, "heldout_cap": 3, "min_heldout": 3, "no_qwen": false, "repair_max_new_tokens": 420, "repair_rounds": 1, "repair_variants": "minimal,broaden,conditional", "run_name": "smoke_qwen_5_helpers", "run_shuffled": true, "sample_seed": 20260627, "task_limit": 5, "train_n": 4, "variants": "monolithic,helpers,robust"}`
- Tasks: `5`
- Direct row-by-row full-task exact: `60.0%`
- Recursive selected-program full-task exact: `20.0%`
- Recursive train-pass rate: `80.0%`
- Recursive oracle among train-passing candidates: `20.0%`

## Run `main_v1`

- Time UTC: `2026-06-27T11:32:03.844775+00:00`
- Elapsed seconds: `1444.3`
- Config: `{"answer_max_new_tokens": 56, "batch_max_new_tokens": 300, "code_max_new_tokens": 420, "heldout_cap": 6, "min_heldout": 3, "no_qwen": false, "repair_max_new_tokens": 420, "repair_rounds": 1, "repair_variants": "minimal,broaden,conditional", "run_name": "main_v1", "run_shuffled": true, "sample_seed": 20260627, "task_limit": 25, "train_n": 4, "variants": "monolithic,robust"}`
- Tasks: `25`
- Direct row-by-row full-task exact: `56.0%`
- Recursive selected-program full-task exact: `40.0%`
- Recursive gated-direct full-task exact: `64.0%`
- Recursive train-pass rate: `52.0%`
- Recursive oracle among train-passing candidates: `40.0%`
