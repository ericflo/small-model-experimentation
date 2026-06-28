# Experiment Log

## Setup

- Created a fresh standalone experiment directory.
- Copied the public transformation benchmark mirror into `/workspace/large_artifacts/qwen_batched_transduction_consistency/prose-benchmarks`.
- Primary metric: full-task exact on held-out rows. A task counts only if every held-out row is exact.
- Secondary metrics: row exact, parse success, and parsed item count.


## Run `smoke_v1`

- Time UTC: `2026-06-27T06:16:05.586794+00:00`
- Elapsed seconds: `12.4`
- Config: `{"batch_max_new_tokens": 160, "heldout_cap": 3, "methods": "row_by_row,batch_all", "min_heldout": 3, "qwen_task_limit": 4, "row_max_new_tokens": 48, "sample_seed": 20260627, "suite": "smoke_v1", "task_limit": 0, "train_n": 4}`
- `batch_all`: row exact 91.7%; full-task exact 75.0%; parse ok 100.0%
- `row_by_row`: row exact 83.3%; full-task exact 75.0%; parse ok 100.0%

## Run `main_v1`

- Time UTC: `2026-06-27T06:25:27.762033+00:00`
- Elapsed seconds: `530.3`
- Config: `{"batch_max_new_tokens": 320, "heldout_cap": 6, "methods": "", "min_heldout": 3, "qwen_task_limit": 40, "row_max_new_tokens": 64, "sample_seed": 20260627, "suite": "main_v1", "task_limit": 0, "train_n": 4}`
- `batch_2`: row exact 70.2%; full-task exact 40.0%; parse ok 99.2%
- `batch_4`: row exact 71.7%; full-task exact 40.0%; parse ok 99.2%
- `batch_all`: row exact 72.5%; full-task exact 45.0%; parse ok 100.0%
- `batch_all_rule_hint`: row exact 69.0%; full-task exact 45.0%; parse ok 100.0%
- `batch_all_shuffled`: row exact 65.8%; full-task exact 42.5%; parse ok 100.0%
- `row_by_row`: row exact 72.1%; full-task exact 50.0%; parse ok 100.0%

## Run `main_v2_prompt_iteration`

- Time UTC: `2026-06-27T06:32:43.777202+00:00`
- Elapsed seconds: `357.6`
- Config: `{"batch_max_new_tokens": 360, "heldout_cap": 6, "methods": "row_by_row,batch_all,batch_all_verify_hint,batch_all_structured", "min_heldout": 3, "qwen_task_limit": 40, "row_max_new_tokens": 64, "sample_seed": 20260627, "suite": "main_v2_prompt_iteration", "task_limit": 0, "train_n": 4}`
- `batch_all`: row exact 72.5%; full-task exact 45.0%; parse ok 100.0%
- `batch_all_structured`: row exact 67.5%; full-task exact 42.5%; parse ok 100.0%
- `batch_all_verify_hint`: row exact 70.0%; full-task exact 45.0%; parse ok 100.0%
- `row_by_row`: row exact 72.1%; full-task exact 50.0%; parse ok 100.0%
