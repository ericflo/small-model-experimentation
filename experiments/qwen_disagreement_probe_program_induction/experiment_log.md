# Experiment Log

## Setup

- Created a fresh standalone experiment directory.
- Large artifacts are kept under `/workspace/large_artifacts/qwen_disagreement_probe_program_induction`.
- Primary metric: strict full-task exact on held-out rows.
- Secondary metrics: row exact, generated-program train-pass rate, probe-label usefulness, and candidate-selection flips.


## Run `smoke_no_qwen`

- Time UTC: `2026-06-27T18:05:55.692908+00:00`
- Elapsed seconds: `0.8`
- Config: `{"answer_max_new_tokens": 64, "batch_max_new_tokens": 320, "code_max_new_tokens": 520, "heldout_cap": 3, "max_probe_pool": 40, "min_heldout": 3, "no_qwen": true, "probe_count": 3, "probe_label_variants": "plain", "probe_score_min": 0.75, "repair_max_new_tokens": 560, "repair_rounds": 1, "repair_variants": "minimal,broaden,conditional", "run_name": "smoke_no_qwen", "run_shuffled": true, "sample_seed": 20260627, "task_limit": 3, "train_n": 4, "variants": "monolithic,robust"}`
- Tasks: `3`
- Direct row-by-row full-task exact: `0.0%`
- Recursive selected-program full-task exact: `0.0%`
- Recursive gated-direct full-task exact: `0.0%`
- Disagreement-probe gated-direct full-task exact: `0.0%`
- Recursive train-pass rate: `0.0%`
- Recursive oracle among train-passing candidates: `0.0%`

## Run `smoke_qwen_5`

- Time UTC: `2026-06-27T18:10:38.193113+00:00`
- Elapsed seconds: `255.2`
- Config: `{"answer_max_new_tokens": 48, "batch_max_new_tokens": 220, "code_max_new_tokens": 420, "heldout_cap": 3, "max_probe_pool": 40, "min_heldout": 3, "no_qwen": false, "probe_count": 3, "probe_label_variants": "plain", "probe_score_min": 0.67, "repair_max_new_tokens": 420, "repair_rounds": 1, "repair_variants": "minimal,broaden,conditional", "run_name": "smoke_qwen_5", "run_shuffled": true, "sample_seed": 20260627, "task_limit": 5, "train_n": 4, "variants": "monolithic,robust"}`
- Tasks: `5`
- Direct row-by-row full-task exact: `60.0%`
- Recursive selected-program full-task exact: `20.0%`
- Recursive gated-direct full-task exact: `60.0%`
- Disagreement-probe gated-direct full-task exact: `60.0%`
- Recursive train-pass rate: `80.0%`
- Recursive oracle among train-passing candidates: `20.0%`

## Run `main_v1`

- Time UTC: `2026-06-27T18:37:15.412379+00:00`
- Elapsed seconds: `1556.2`
- Config: `{"answer_max_new_tokens": 56, "batch_max_new_tokens": 300, "code_max_new_tokens": 420, "heldout_cap": 6, "max_probe_pool": 40, "min_heldout": 3, "no_qwen": false, "probe_count": 4, "probe_label_variants": "plain,consistency", "probe_score_min": 0.67, "repair_max_new_tokens": 420, "repair_rounds": 1, "repair_variants": "minimal,broaden,conditional", "run_name": "main_v1", "run_shuffled": true, "sample_seed": 20260627, "task_limit": 25, "train_n": 4, "variants": "monolithic,robust"}`
- Tasks: `25`
- Direct row-by-row full-task exact: `56.0%`
- Recursive selected-program full-task exact: `40.0%`
- Recursive gated-direct full-task exact: `64.0%`
- Disagreement-probe gated-direct full-task exact: `64.0%`
- Recursive train-pass rate: `52.0%`
- Recursive oracle among train-passing candidates: `40.0%`
- Interpretation: disagreement probes did not add measurable selection power. Probe-gated, non-probe gated, and random-probe gated selection all reached `64.0%`; shuffled probe labels fell back to the direct baseline at `56.0%`. Probe-gated and non-probe gated outcomes differed on `0/25` tasks, so the deployable gain came from conservative program/direct gating rather than probe-based discrimination.
