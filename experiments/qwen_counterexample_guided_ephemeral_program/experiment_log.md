# Experiment Log

## Setup

- Created a fresh standalone experiment directory.
- Large artifacts are kept under `/workspace/large_artifacts/qwen_counterexample_guided_ephemeral_program`.
- Primary metric: strict full-task exact on held-out rows.
- Intervention: generate task-local executable programs, create synthetic disagreement probes, label probes with the model, and select or route programs using the expanded label set.

## Run `smoke_no_qwen`

- Time UTC: `2026-06-28T03:23:44.477998+00:00`
- Elapsed seconds: `1.0`
- Config: `{"answer_max_new_tokens": 64, "batch_max_new_tokens": 220, "code_max_new_tokens": 620, "gate_probe_score": 0.75, "heldout_cap": 3, "min_confident_probes": 2, "min_heldout": 3, "min_probe_consensus": 0.6666666666666666, "no_qwen": true, "probe_count": 2, "probe_label_styles": "plain,rule,strict", "run_name": "smoke_no_qwen", "seed": 20260628, "task_limit": 3, "train_n": 3, "variants": "direct,helpers,regex,conditional"}`
- Tasks: `3`
- `direct_qwen_row`: `0.0%` full-task exact.
- `program_visible`: `0.0%` full-task exact.
- `program_ceg`: `0.0%` full-task exact.
- `program_ceg_gated`: `0.0%` full-task exact.
- `program_ceg_router`: `0.0%` full-task exact.
- `program_shuffled_probe_labels`: `0.0%` full-task exact.
- Hidden candidate oracle: `0.0%` full-task exact.

## Run `smoke_qwen_3`

- Time UTC: `2026-06-28T03:26:45.006601+00:00`
- Elapsed seconds: `140.2`
- Config: `{"answer_max_new_tokens": 56, "batch_max_new_tokens": 180, "code_max_new_tokens": 520, "gate_probe_score": 0.75, "heldout_cap": 3, "min_confident_probes": 2, "min_heldout": 3, "min_probe_consensus": 0.6666666666666666, "no_qwen": false, "probe_count": 2, "probe_label_styles": "plain,rule,strict", "run_name": "smoke_qwen_3", "seed": 20260628, "task_limit": 3, "train_n": 4, "variants": "direct,helpers,regex,conditional"}`
- Tasks: `3`
- `direct_qwen_row`: `66.7%` full-task exact.
- `program_visible`: `33.3%` full-task exact.
- `program_ceg`: `33.3%` full-task exact.
- `program_ceg_gated`: `66.7%` full-task exact.
- `program_ceg_router`: `66.7%` full-task exact.
- `program_shuffled_probe_labels`: `33.3%` full-task exact.
- Hidden candidate oracle: `33.3%` full-task exact.

## Run `smoke_qwen_3_repair`

- Time UTC: `2026-06-28T03:32:20.510640+00:00`
- Elapsed seconds: `230.7`
- Config: `{"answer_max_new_tokens": 56, "batch_max_new_tokens": 180, "code_max_new_tokens": 520, "gate_probe_score": 0.75, "heldout_cap": 3, "min_confident_probes": 2, "min_heldout": 3, "min_probe_consensus": 0.6666666666666666, "no_qwen": false, "probe_count": 2, "probe_label_styles": "plain,rule,strict", "repair_rounds": 1, "run_name": "smoke_qwen_3_repair", "seed": 20260628, "task_limit": 3, "train_n": 4, "variants": "direct,helpers,regex,conditional"}`
- Tasks: `3`
- `direct_qwen_row`: `66.7%` full-task exact.
- `program_visible`: `33.3%` full-task exact.
- `program_ceg`: `33.3%` full-task exact.
- `program_ceg_gated`: `66.7%` full-task exact.
- `program_ceg_router`: `66.7%` full-task exact.
- `program_shuffled_probe_labels`: `33.3%` full-task exact.
- Hidden candidate oracle: `33.3%` full-task exact.

## Run `main_v1`

- Time UTC: `2026-06-28T03:56:44.332833+00:00`
- Elapsed seconds: `1410.9`
- Config: `{"answer_max_new_tokens": 56, "batch_max_new_tokens": 220, "code_max_new_tokens": 560, "gate_probe_score": 0.75, "heldout_cap": 4, "min_confident_probes": 2, "min_heldout": 3, "min_probe_consensus": 0.6666666666666666, "no_qwen": false, "probe_count": 4, "probe_label_styles": "plain,rule,strict", "repair_rounds": 1, "run_name": "main_v1", "seed": 20260628, "task_limit": 24, "train_n": 4, "variants": "direct,helpers,regex,conditional"}`
- Tasks: `24`
- `direct_qwen_row`: `75.0%` full-task exact.
- `program_visible`: `37.5%` full-task exact.
- `program_ceg`: `41.7%` full-task exact.
- `program_ceg_gated`: `75.0%` full-task exact.
- `program_ceg_router`: `75.0%` full-task exact.
- `program_shuffled_probe_labels`: `41.7%` full-task exact.
- Hidden candidate oracle: `41.7%` full-task exact.
