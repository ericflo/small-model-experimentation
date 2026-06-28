# Experiment Log

## Setup

- Created fresh standalone experiment directory: `/workspace/experiments/qwen_counterfactual_episodic_icl`.
- Large artifacts directory: `/workspace/large_artifacts/qwen_counterfactual_episodic_icl`.
- Public benchmark mirror is stored under the large-artifact directory.
- Primary question: can counterfactual episodic posttraining improve sparse-example task induction rather than merely memorizing a family prior?

## Run `smoke_no_train`

- Time UTC: `2026-06-28T04:16:45.068880+00:00`
- Config: `{"batch_size": 1, "grad_accum": 8, "log_every": 10, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 32, "model": "Qwen/Qwen3-4B", "no_train": true, "public_heldout_n": 3, "public_task_limit": 1, "query_n": 2, "run_name": "smoke_no_train", "seed": 20260628, "smoke": true, "support_n": 4, "synthetic_eval_pairs": 1, "train_episodes": 16, "train_rows_per_episode": 2, "train_steps": 1, "warmup_steps": 20, "weight_decay": 0.0}`
- Elapsed seconds: `36.3`
- Synthetic adapter full-task exact: `0.0%`
- Public adapter full-task exact: `0.0%`
- Report: `/workspace/experiments/qwen_counterfactual_episodic_icl/reports/qwen_counterfactual_episodic_icl_report.md`

## Run `smoke_train`

- Time UTC: `2026-06-28T04:18:04.969847+00:00`
- Config: `{"batch_size": 1, "grad_accum": 2, "log_every": 1, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 32, "model": "Qwen/Qwen3-4B", "no_train": false, "public_heldout_n": 3, "public_task_limit": 1, "query_n": 2, "run_name": "smoke_train", "seed": 20260628, "smoke": true, "support_n": 4, "synthetic_eval_pairs": 1, "train_episodes": 16, "train_rows_per_episode": 2, "train_steps": 2, "warmup_steps": 20, "weight_decay": 0.0}`
- Elapsed seconds: `60.1`
- Synthetic adapter full-task exact: `0.0%`
- Public adapter full-task exact: `0.0%`
- Report: `/workspace/experiments/qwen_counterfactual_episodic_icl/reports/qwen_counterfactual_episodic_icl_report.md`

## Run `pilot_v1`

- Time UTC: `2026-06-28T04:20:11.971541+00:00`
- Config: `{"batch_size": 1, "grad_accum": 4, "log_every": 5, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 24, "model": "Qwen/Qwen3-4B", "no_train": false, "public_heldout_n": 3, "public_task_limit": 8, "query_n": 2, "run_name": "pilot_v1", "seed": 20260628, "smoke": false, "support_n": 4, "synthetic_eval_pairs": 8, "train_episodes": 240, "train_rows_per_episode": 2, "train_steps": 50, "warmup_steps": 20, "weight_decay": 0.0}`
- Elapsed seconds: `174.7`
- Synthetic adapter full-task exact: `93.8%`
- Public adapter full-task exact: `37.5%`
- Report: `/workspace/experiments/qwen_counterfactual_episodic_icl/reports/qwen_counterfactual_episodic_icl_report.md`

## Run `main_v1`

- Time UTC: `2026-06-28T04:23:25.047618+00:00`
- Config: `{"batch_size": 1, "grad_accum": 4, "log_every": 10, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 24, "model": "Qwen/Qwen3-4B", "no_train": false, "public_heldout_n": 3, "public_task_limit": 30, "query_n": 2, "run_name": "main_v1", "seed": 20260628, "smoke": false, "support_n": 4, "synthetic_eval_pairs": 30, "train_episodes": 800, "train_rows_per_episode": 2, "train_steps": 120, "warmup_steps": 20, "weight_decay": 0.0}`
- `main_v1` was interrupted before model load after a boundedness bug was found in counterfactual rule-pair construction. No result artifacts were produced for that run. The generator was patched to discard infeasible rule pairs after bounded search.

## Run `main_v2`

- Time UTC: `2026-06-28T04:27:24.417084+00:00`
- Config: `{"batch_size": 1, "grad_accum": 4, "log_every": 10, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 24, "model": "Qwen/Qwen3-4B", "no_train": false, "public_heldout_n": 3, "public_task_limit": 30, "query_n": 2, "run_name": "main_v2", "seed": 20260628, "smoke": false, "support_n": 4, "synthetic_eval_pairs": 30, "train_episodes": 800, "train_rows_per_episode": 2, "train_steps": 120, "warmup_steps": 20, "weight_decay": 0.0}`
- Elapsed seconds: `588.9`
- Synthetic adapter full-task exact: `91.7%`
- Public adapter full-task exact: `56.7%`
- Report: `/workspace/experiments/qwen_counterfactual_episodic_icl/reports/qwen_counterfactual_episodic_icl_report.md`
