# Experiment Log

## Setup

- Created fresh standalone experiment directory: `/workspace/experiments/qwen_support_contrastive_meta_icl`.
- Large artifacts directory: `/workspace/large_artifacts/qwen_support_contrastive_meta_icl`.
- Primary question: does an explicit support-contrastive loss improve public text-transformation accuracy while making performance depend more causally on intact support examples?
- Planned process: implement harness, run smoke, run pilot, run main fixed-eval matrix, aggregate with Markdown and HTML reports.

## Smoke `smoke_contrastive`

- Contrastive training path compiled and ran for two optimizer steps.
- Multi-forward margin loss did not OOM.
- Smoke aggregation produced Markdown and HTML reports.
- Patched shuffled-support eval seeding to use a deterministic support-mode offset.

## Pilot

- `pilot_contrastive_cf`: 18 public tasks, 24 synthetic counterfactual tasks, 40 optimizer steps.
  - Public full-task exact: normal `55.6%`, shuffled `5.6%`, no support `0.0%`.
  - Synthetic full-task exact: normal `79.2%`, shuffled `12.5%`, contrast support `0.0%`, no support `0.0%`.
- `pilot_ce_cf`: same fixed eval, CE-only counterfactual control, 40 optimizer steps.
  - Public full-task exact: normal `50.0%`, shuffled `27.8%`, no support `5.6%`.
  - Synthetic full-task exact: normal `83.3%`, shuffled `41.7%`, contrast support `0.0%`.
- Pilot read: the support-contrastive objective did not improve normal public accuracy over CE in the pilot, but it produced a much cleaner corrupted-support gap. Proceeding to the full fixed-eval matrix.

## Main Matrix

- Completed:
  - `main_contrastive_s1`, `main_contrastive_s2`, `main_contrastive_s3`
  - `main_ce_cf_s1`
  - `main_ce_ordinary_s1`
  - `main_ce_shuffled_labels_s1`
- Fixed evaluation:
  - 45 public PROSE tasks, 3 held-out rows per task.
  - 60 synthetic counterfactual tasks, 2 held-out rows per task.
- Public full-task exact:
  - Base: `17.8%`.
  - Support-contrastive mean: `50.4%`, std `1.3%`.
  - CE counterfactual: `53.3%`.
  - CE ordinary: `55.6%`.
  - CE shuffled-label: `48.9%`.
- Public corrupted-support controls:
  - Support-contrastive: normal `50.4%`, shuffled `7.4%`, no-support `0.0%`.
  - CE counterfactual: normal `53.3%`, shuffled `33.3%`, no-support `4.4%`.
  - CE ordinary: normal `55.6%`, shuffled `22.2%`, no-support `4.4%`.
  - CE shuffled-label: normal `48.9%`, shuffled `48.9%`, no-support `2.2%`.
- Synthetic corrupted-support controls:
  - Support-contrastive: normal `83.9%`, shuffled `6.7%`, contrast-support `0.0%`, no-support `0.0%`.
  - CE shuffled-label: normal `91.7%`, shuffled `83.3%`.
- Main read: the support-contrastive objective does what it was designed to do: it makes performance causally depend on intact support examples. It is not the best raw public-accuracy recipe at this margin/weight, because CE-only ordinary training reaches higher normal-support accuracy. The live next question is tuning or scheduling the margin so it preserves CE accuracy while keeping the corruption gap.
- Final Markdown report: `/workspace/experiments/qwen_support_contrastive_meta_icl/reports/qwen_support_contrastive_meta_icl_report.md`
- Final HTML report: `/workspace/experiments/qwen_support_contrastive_meta_icl/reports/qwen_support_contrastive_meta_icl_report.html`

## Run `smoke_contrastive`

- Time UTC: `2026-06-28T16:41:31.417074+00:00`
- Config: `{"contrast_weight": 0.35, "eval_seed": 20260702, "grad_accum": 2, "log_every": 1, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "margin": 0.75, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 24, "model": "Qwen/Qwen3-4B", "no_train": false, "objective": "support_contrastive", "public_heldout_n": 3, "public_task_limit": 2, "query_n": 2, "run_name": "smoke_contrastive", "seed": 20260628, "skip_base_eval": false, "smoke": true, "support_n": 4, "synthetic_eval_pairs": 2, "train_episodes": 16, "train_mode": "counterfactual", "train_rows_per_episode": 2, "train_steps": 2, "warmup_steps": 20, "weight_decay": 0.0}`
- Elapsed seconds: `107.7`
- Summary rows: `14`
- Latest run report: `/workspace/experiments/qwen_support_contrastive_meta_icl/reports/qwen_support_contrastive_meta_icl_latest_run_report.md`

## Run `pilot_contrastive_cf`

- Time UTC: `2026-06-28T16:45:45.172706+00:00`
- Config: `{"contrast_weight": 0.35, "eval_seed": 20260702, "grad_accum": 4, "log_every": 5, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "margin": 0.75, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 24, "model": "Qwen/Qwen3-4B", "no_train": false, "objective": "support_contrastive", "public_heldout_n": 3, "public_task_limit": 18, "query_n": 2, "run_name": "pilot_contrastive_cf", "seed": 20260628, "skip_base_eval": false, "smoke": false, "support_n": 4, "synthetic_eval_pairs": 12, "train_episodes": 300, "train_mode": "counterfactual", "train_rows_per_episode": 2, "train_steps": 40, "warmup_steps": 20, "weight_decay": 0.0}`
- Elapsed seconds: `564.0`
- Summary rows: `14`
- Latest run report: `/workspace/experiments/qwen_support_contrastive_meta_icl/reports/qwen_support_contrastive_meta_icl_latest_run_report.md`

## Run `pilot_ce_cf`

- Time UTC: `2026-06-28T16:55:23.733550+00:00`
- Config: `{"contrast_weight": 0.35, "eval_seed": 20260702, "grad_accum": 4, "log_every": 5, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "margin": 0.75, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 24, "model": "Qwen/Qwen3-4B", "no_train": false, "objective": "ce", "public_heldout_n": 3, "public_task_limit": 18, "query_n": 2, "run_name": "pilot_ce_cf", "seed": 20260628, "skip_base_eval": true, "smoke": false, "support_n": 4, "synthetic_eval_pairs": 12, "train_episodes": 300, "train_mode": "counterfactual", "train_rows_per_episode": 2, "train_steps": 40, "warmup_steps": 20, "weight_decay": 0.0}`
- Elapsed seconds: `171.4`
- Summary rows: `7`
- Latest run report: `/workspace/experiments/qwen_support_contrastive_meta_icl/reports/qwen_support_contrastive_meta_icl_latest_run_report.md`

## Run `main_contrastive_s1`

- Time UTC: `2026-06-28T16:59:13.158687+00:00`
- Config: `{"contrast_weight": 0.35, "eval_seed": 20260702, "grad_accum": 4, "log_every": 10, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "margin": 0.75, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 24, "model": "Qwen/Qwen3-4B", "no_train": false, "objective": "support_contrastive", "public_heldout_n": 3, "public_task_limit": 45, "query_n": 2, "run_name": "main_contrastive_s1", "seed": 20260628, "skip_base_eval": false, "smoke": false, "support_n": 4, "synthetic_eval_pairs": 30, "train_episodes": 700, "train_mode": "counterfactual", "train_rows_per_episode": 2, "train_steps": 80, "warmup_steps": 20, "weight_decay": 0.0}`
- Elapsed seconds: `1135.6`
- Summary rows: `14`
- Latest run report: `/workspace/experiments/qwen_support_contrastive_meta_icl/reports/qwen_support_contrastive_meta_icl_latest_run_report.md`

## Run `main_contrastive_s2`

- Time UTC: `2026-06-28T17:18:31.066226+00:00`
- Config: `{"contrast_weight": 0.35, "eval_seed": 20260702, "grad_accum": 4, "log_every": 10, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "margin": 0.75, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 24, "model": "Qwen/Qwen3-4B", "no_train": false, "objective": "support_contrastive", "public_heldout_n": 3, "public_task_limit": 45, "query_n": 2, "run_name": "main_contrastive_s2", "seed": 20260629, "skip_base_eval": true, "smoke": false, "support_n": 4, "synthetic_eval_pairs": 30, "train_episodes": 700, "train_mode": "counterfactual", "train_rows_per_episode": 2, "train_steps": 80, "warmup_steps": 20, "weight_decay": 0.0}`
- Elapsed seconds: `754.6`
- Summary rows: `7`
- Latest run report: `/workspace/experiments/qwen_support_contrastive_meta_icl/reports/qwen_support_contrastive_meta_icl_latest_run_report.md`

## Run `main_contrastive_s3`

- Time UTC: `2026-06-28T17:31:23.436980+00:00`
- Config: `{"contrast_weight": 0.35, "eval_seed": 20260702, "grad_accum": 4, "log_every": 10, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "margin": 0.75, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 24, "model": "Qwen/Qwen3-4B", "no_train": false, "objective": "support_contrastive", "public_heldout_n": 3, "public_task_limit": 45, "query_n": 2, "run_name": "main_contrastive_s3", "seed": 20260630, "skip_base_eval": true, "smoke": false, "support_n": 4, "synthetic_eval_pairs": 30, "train_episodes": 700, "train_mode": "counterfactual", "train_rows_per_episode": 2, "train_steps": 80, "warmup_steps": 20, "weight_decay": 0.0}`
- Elapsed seconds: `760.6`
- Summary rows: `7`
- Latest run report: `/workspace/experiments/qwen_support_contrastive_meta_icl/reports/qwen_support_contrastive_meta_icl_latest_run_report.md`

## Run `main_ce_cf_s1`

- Time UTC: `2026-06-28T17:44:32.745451+00:00`
- Config: `{"contrast_weight": 0.35, "eval_seed": 20260702, "grad_accum": 4, "log_every": 10, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "margin": 0.75, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 24, "model": "Qwen/Qwen3-4B", "no_train": false, "objective": "ce", "public_heldout_n": 3, "public_task_limit": 45, "query_n": 2, "run_name": "main_ce_cf_s1", "seed": 20260628, "skip_base_eval": true, "smoke": false, "support_n": 4, "synthetic_eval_pairs": 30, "train_episodes": 700, "train_mode": "counterfactual", "train_rows_per_episode": 2, "train_steps": 80, "warmup_steps": 20, "weight_decay": 0.0}`
- Elapsed seconds: `410.6`
- Summary rows: `7`
- Latest run report: `/workspace/experiments/qwen_support_contrastive_meta_icl/reports/qwen_support_contrastive_meta_icl_latest_run_report.md`

## Run `main_ce_ordinary_s1`

- Time UTC: `2026-06-28T17:51:37.497945+00:00`
- Config: `{"contrast_weight": 0.35, "eval_seed": 20260702, "grad_accum": 4, "log_every": 10, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "margin": 0.75, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 24, "model": "Qwen/Qwen3-4B", "no_train": false, "objective": "ce", "public_heldout_n": 3, "public_task_limit": 45, "query_n": 2, "run_name": "main_ce_ordinary_s1", "seed": 20260628, "skip_base_eval": true, "smoke": false, "support_n": 4, "synthetic_eval_pairs": 30, "train_episodes": 700, "train_mode": "ordinary", "train_rows_per_episode": 2, "train_steps": 80, "warmup_steps": 20, "weight_decay": 0.0}`
- Elapsed seconds: `374.7`
- Summary rows: `7`
- Latest run report: `/workspace/experiments/qwen_support_contrastive_meta_icl/reports/qwen_support_contrastive_meta_icl_latest_run_report.md`

## Run `main_ce_shuffled_labels_s1`

- Time UTC: `2026-06-28T17:58:05.949981+00:00`
- Config: `{"contrast_weight": 0.35, "eval_seed": 20260702, "grad_accum": 4, "log_every": 10, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "margin": 0.75, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 24, "model": "Qwen/Qwen3-4B", "no_train": false, "objective": "ce", "public_heldout_n": 3, "public_task_limit": 45, "query_n": 2, "run_name": "main_ce_shuffled_labels_s1", "seed": 20260628, "skip_base_eval": true, "smoke": false, "support_n": 4, "synthetic_eval_pairs": 30, "train_episodes": 700, "train_mode": "shuffled_labels", "train_rows_per_episode": 2, "train_steps": 80, "warmup_steps": 20, "weight_decay": 0.0}`
- Elapsed seconds: `397.1`
- Summary rows: `7`
- Latest run report: `/workspace/experiments/qwen_support_contrastive_meta_icl/reports/qwen_support_contrastive_meta_icl_latest_run_report.md`
