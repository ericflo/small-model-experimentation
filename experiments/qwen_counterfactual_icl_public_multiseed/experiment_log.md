# Experiment Log

## Setup

- Created fresh standalone experiment directory: `/workspace/experiments/qwen_counterfactual_icl_public_multiseed`.
- Large artifacts directory: `/workspace/large_artifacts/qwen_counterfactual_icl_public_multiseed`.
- Public benchmark mirror copied under the large-artifact directory.
- Primary question: does counterfactual episodic LoRA posttraining improve support-conditioned public text transformation accuracy across seeds, beyond output-format or generic synthetic-training effects?

## Run `smoke_cf`

- Time UTC: `2026-06-28T05:50:51.863838+00:00`
- Config: `{"batch_size": 1, "eval_seed": 20260701, "grad_accum": 2, "log_every": 1, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 24, "model": "Qwen/Qwen3-4B", "no_train": false, "public_heldout_n": 3, "public_task_limit": 2, "query_n": 2, "run_name": "smoke_cf", "seed": 20260628, "skip_base_eval": false, "smoke": true, "support_n": 4, "synthetic_eval_pairs": 2, "train_episodes": 16, "train_mode": "counterfactual", "train_rows_per_episode": 2, "train_steps": 2, "warmup_steps": 20, "weight_decay": 0.0}`
- Elapsed seconds: `80.8`
- Synthetic adapter full-task exact: `25.0%`
- Public adapter full-task exact: `0.0%`
- Report: `/workspace/experiments/qwen_counterfactual_icl_public_multiseed/reports/qwen_counterfactual_icl_public_multiseed_report.md`

## Aggregation Harness

- Added `src/aggregate_multiseed.py`.
- Compile check passed for the training and aggregate scripts.
- Smoke aggregation passed against `smoke_cf`.
- Main gate plan: three counterfactual seeds plus ordinary-training and shuffled-label controls, all with fixed public/synthetic eval seed.

## Main Gate Results

- Completed `cf_s1`, `cf_s2`, `cf_s3`, `ordinary_s1`, and `shuffled_train_s1`.
- Fixed eval set: 60 synthetic counterfactual tasks and 45 public PROSE tasks.
- Base public full-task exact: `20.0%`.
- Counterfactual adapter public full-task exact: mean `60.7%`, std `2.6%` over 3 seeds.
- Counterfactual adapter public controls: shuffled support `28.1%`, no support `1.5%`.
- Ordinary synthetic-training control public full-task exact: `64.4%`.
- Shuffled-label training control public full-task exact: `62.2%`; shuffled support remains high at `51.1%`.
- Main read: the public improvement is stable, but it is not uniquely explained by counterfactual support-induction training. Generic synthetic ICL tuning and even corrupted-support training reproduce much of the public gain, so the effect is likely a broader prompt-format/task-family adaptation plus some support sensitivity.
- Final Markdown report: `/workspace/experiments/qwen_counterfactual_icl_public_multiseed/reports/qwen_counterfactual_icl_public_multiseed_report.md`
- Final HTML report: `/workspace/experiments/qwen_counterfactual_icl_public_multiseed/reports/qwen_counterfactual_icl_public_multiseed_report.html`

## Run `cf_s1`

- Time UTC: `2026-06-28T05:56:28.260234+00:00`
- Config: `{"batch_size": 1, "eval_seed": 20260701, "grad_accum": 4, "log_every": 10, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 24, "model": "Qwen/Qwen3-4B", "no_train": false, "public_heldout_n": 3, "public_task_limit": 45, "query_n": 2, "run_name": "cf_s1", "seed": 20260628, "skip_base_eval": false, "smoke": false, "support_n": 4, "synthetic_eval_pairs": 30, "train_episodes": 800, "train_mode": "counterfactual", "train_rows_per_episode": 2, "train_steps": 100, "warmup_steps": 20, "weight_decay": 0.0}`
- Elapsed seconds: `729.1`
- Synthetic adapter full-task exact: `91.7%`
- Public adapter full-task exact: `57.8%`
- Report: `/workspace/experiments/qwen_counterfactual_icl_public_multiseed/reports/qwen_counterfactual_icl_public_multiseed_report.md`

## Run `cf_s2`

- Time UTC: `2026-06-28T06:08:54.410400+00:00`
- Config: `{"batch_size": 1, "eval_seed": 20260701, "grad_accum": 4, "log_every": 10, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 24, "model": "Qwen/Qwen3-4B", "no_train": false, "public_heldout_n": 3, "public_task_limit": 45, "query_n": 2, "run_name": "cf_s2", "seed": 20260629, "skip_base_eval": true, "smoke": false, "support_n": 4, "synthetic_eval_pairs": 30, "train_episodes": 800, "train_mode": "counterfactual", "train_rows_per_episode": 2, "train_steps": 100, "warmup_steps": 20, "weight_decay": 0.0}`
- Elapsed seconds: `409.3`
- Synthetic adapter full-task exact: `95.0%`
- Public adapter full-task exact: `62.2%`
- Report: `/workspace/experiments/qwen_counterfactual_icl_public_multiseed/reports/qwen_counterfactual_icl_public_multiseed_report.md`

## Run `cf_s3`

- Time UTC: `2026-06-28T06:16:00.820755+00:00`
- Config: `{"batch_size": 1, "eval_seed": 20260701, "grad_accum": 4, "log_every": 10, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 24, "model": "Qwen/Qwen3-4B", "no_train": false, "public_heldout_n": 3, "public_task_limit": 45, "query_n": 2, "run_name": "cf_s3", "seed": 20260630, "skip_base_eval": true, "smoke": false, "support_n": 4, "synthetic_eval_pairs": 30, "train_episodes": 800, "train_mode": "counterfactual", "train_rows_per_episode": 2, "train_steps": 100, "warmup_steps": 20, "weight_decay": 0.0}`
- Elapsed seconds: `404.6`
- Synthetic adapter full-task exact: `91.7%`
- Public adapter full-task exact: `62.2%`
- Report: `/workspace/experiments/qwen_counterfactual_icl_public_multiseed/reports/qwen_counterfactual_icl_public_multiseed_report.md`

## Run `ordinary_s1`

- Time UTC: `2026-06-28T06:22:59.909062+00:00`
- Config: `{"batch_size": 1, "eval_seed": 20260701, "grad_accum": 4, "log_every": 10, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 24, "model": "Qwen/Qwen3-4B", "no_train": false, "public_heldout_n": 3, "public_task_limit": 45, "query_n": 2, "run_name": "ordinary_s1", "seed": 20260628, "skip_base_eval": true, "smoke": false, "support_n": 4, "synthetic_eval_pairs": 30, "train_episodes": 800, "train_mode": "ordinary", "train_rows_per_episode": 2, "train_steps": 100, "warmup_steps": 20, "weight_decay": 0.0}`
- Elapsed seconds: `408.6`
- Synthetic adapter full-task exact: `95.0%`
- Public adapter full-task exact: `64.4%`
- Report: `/workspace/experiments/qwen_counterfactual_icl_public_multiseed/reports/qwen_counterfactual_icl_public_multiseed_report.md`

## Run `shuffled_train_s1`

- Time UTC: `2026-06-28T06:30:03.483561+00:00`
- Config: `{"batch_size": 1, "eval_seed": 20260701, "grad_accum": 4, "log_every": 10, "lora_alpha": 32, "lora_dropout": 0.05, "lora_r": 16, "lr": 0.0002, "max_grad_norm": 1.0, "max_length": 768, "max_new_tokens": 24, "model": "Qwen/Qwen3-4B", "no_train": false, "public_heldout_n": 3, "public_task_limit": 45, "query_n": 2, "run_name": "shuffled_train_s1", "seed": 20260628, "skip_base_eval": true, "smoke": false, "support_n": 4, "synthetic_eval_pairs": 30, "train_episodes": 800, "train_mode": "shuffled_labels", "train_rows_per_episode": 2, "train_steps": 100, "warmup_steps": 20, "weight_decay": 0.0}`
- Elapsed seconds: `371.2`
- Synthetic adapter full-task exact: `95.0%`
- Public adapter full-task exact: `62.2%`
- Report: `/workspace/experiments/qwen_counterfactual_icl_public_multiseed/reports/qwen_counterfactual_icl_public_multiseed_report.md`
