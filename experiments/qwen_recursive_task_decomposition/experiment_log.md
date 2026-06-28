# Experiment Log

## Setup

- Created a fresh standalone experiment directory.
- Target benchmark: public Microsoft PROSE `Transformation.Text` tasks mirrored under `/workspace/large_artifacts/qwen_recursive_task_decomposition/prose-benchmarks`.
- Primary metric: full-task exact on held-out rows. A task counts only if every held-out row is exact.
- Secondary metric: row exact, used only to diagnose task-level consistency.


## Run `smoke_static`

- Time UTC: `2026-06-27T05:12:36.848797+00:00`
- Static tasks: `12`; Qwen tasks: `0`
- Config: `{"answer_max_new_tokens": 64, "child_limit": 4, "heldout_cap": 8, "max_candidates": 8000, "qwen_heldout_cap": 6, "qwen_min_heldout": 3, "qwen_task_limit": 12, "recursive_depth": 2, "rule_max_new_tokens": 180, "run_qwen": false, "sample_seed": 20260627, "suite": "smoke_static", "task_limit": 12, "train_n": 4}`
- Static summary:
  - `static_mono_examples` full-task exact: 33.3%
  - `static_mono_oracle` full-task exact: 50.0%
  - `static_recursive_examples` full-task exact: 33.3%
  - `static_recursive_oracle` full-task exact: 50.0%
  - `static_recursive_shuffled` full-task exact: 0.0%

## Run `smoke_qwen`

- Time UTC: `2026-06-27T05:13:59.780118+00:00`
- Static tasks: `20`; Qwen tasks: `4`
- Config: `{"answer_max_new_tokens": 48, "child_limit": 3, "heldout_cap": 6, "max_candidates": 5000, "qwen_heldout_cap": 3, "qwen_min_heldout": 3, "qwen_task_limit": 4, "recursive_depth": 1, "rule_max_new_tokens": 120, "run_qwen": true, "sample_seed": 20260627, "suite": "smoke_qwen", "task_limit": 20, "train_n": 4}`
- Static summary:
  - `static_mono_examples` full-task exact: 30.0%
  - `static_mono_oracle` full-task exact: 50.0%
  - `static_recursive_examples` full-task exact: 30.0%
  - `static_recursive_oracle` full-task exact: 50.0%
  - `static_recursive_shuffled` full-task exact: 0.0%
- Qwen summary:
  - `direct_qwen` row exact: 75.0%; full-task exact: 50.0%
  - `locked_rule_qwen` row exact: 58.3%; full-task exact: 50.0%
  - `shuffled_rule_qwen` row exact: 58.3%; full-task exact: 25.0%

## Run `main_v1`

- Time UTC: `2026-06-27T05:35:27.915748+00:00`
- Static tasks: `309`; Qwen tasks: `30`
- Config: `{"answer_max_new_tokens": 56, "child_limit": 5, "heldout_cap": 50, "max_candidates": 12000, "qwen_heldout_cap": 6, "qwen_min_heldout": 3, "qwen_task_limit": 30, "recursive_depth": 2, "rule_max_new_tokens": 160, "run_qwen": true, "sample_seed": 20260627, "suite": "main_v1", "task_limit": 0, "train_n": 4}`
- Static summary:
  - `static_mono_examples` full-task exact: 22.7%
  - `static_mono_oracle` full-task exact: 29.4%
  - `static_recursive_examples` full-task exact: 22.7%
  - `static_recursive_oracle` full-task exact: 34.3%
  - `static_recursive_shuffled` full-task exact: 0.3%
- Qwen summary:
  - `direct_qwen` row exact: 69.4%; full-task exact: 46.7%
  - `locked_rule_qwen` row exact: 61.7%; full-task exact: 46.7%
  - `shuffled_rule_qwen` row exact: 53.6%; full-task exact: 36.7%

## Run `syntax_check`

- Time UTC: `2026-06-27T05:36:58.054670+00:00`
- Static tasks: `3`; Qwen tasks: `0`
- Config: `{"answer_max_new_tokens": 64, "child_limit": 2, "heldout_cap": 3, "max_candidates": 1000, "qwen_heldout_cap": 6, "qwen_min_heldout": 3, "qwen_task_limit": 12, "recursive_depth": 1, "rule_max_new_tokens": 180, "rule_styles": "decompose", "run_qwen": false, "sample_seed": 20260627, "suite": "syntax_check", "task_limit": 3, "train_n": 4, "train_verify_rules": false}`
- Static summary:
  - `static_mono_examples` full-task exact: 33.3%
  - `static_mono_oracle` full-task exact: 33.3%
  - `static_recursive_examples` full-task exact: 33.3%
  - `static_recursive_oracle` full-task exact: 33.3%
  - `static_recursive_shuffled` full-task exact: 0.0%

## Run `main_v2_train_verified_rules`

- Time UTC: `2026-06-27T05:49:25.174356+00:00`
- Static tasks: `309`; Qwen tasks: `30`
- Config: `{"answer_max_new_tokens": 56, "child_limit": 5, "heldout_cap": 50, "max_candidates": 12000, "qwen_heldout_cap": 6, "qwen_min_heldout": 3, "qwen_task_limit": 30, "recursive_depth": 2, "reuse_static_from": "main_v1", "rule_max_new_tokens": 150, "rule_styles": "decompose,terse,conditional", "run_qwen": true, "sample_seed": 20260627, "suite": "main_v2_train_verified_rules", "task_limit": 0, "train_n": 4, "train_verify_rules": true}`
- Static summary:
  - `static_mono_examples` full-task exact: 22.7%
  - `static_mono_oracle` full-task exact: 29.4%
  - `static_recursive_examples` full-task exact: 22.7%
  - `static_recursive_oracle` full-task exact: 34.3%
  - `static_recursive_shuffled` full-task exact: 0.3%
- Qwen summary:
  - `direct_qwen` row exact: 69.4%; full-task exact: 46.7%
  - `locked_rule_qwen` row exact: 59.4%; full-task exact: 43.3%
  - `shuffled_rule_qwen` row exact: 58.3%; full-task exact: 33.3%
