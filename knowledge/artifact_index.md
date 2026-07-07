# Artifact Index

This is a repository-level inventory. Each experiment remains the source of truth for its own artifacts.

## Standard Experiment Shape

- `README.md`
- `metadata.yaml`
- `experiment_log.md`
- `src/`
- `scripts/`
- `configs/`
- `data/`
- `runs/`
- `analysis/`
- `reports/`

## Top-Level Directory Coverage

| Directory | Experiments |
| --- | ---: |
| `reports/` | 194 |
| `src/` | 182 |
| `runs/` | 130 |
| `analysis/` | 120 |
| `scripts/` | 103 |
| `configs/` | 100 |
| `data/` | 93 |
| `logs/` | 64 |
| `run_logs/` | 55 |
| `figures/` | 10 |
| `suite_logs/` | 1 |

## File Extensions

| Extension | Files |
| --- | ---: |
| `.csv` | 2134 |
| `.json` | 1973 |
| `.py` | 959 |
| `.png` | 840 |
| `.md` | 773 |
| `.jsonl` | 568 |
| `.log` | 511 |
| `.yaml` | 272 |
| `.html` | 90 |
| `.sh` | 19 |
| `.txt` | 4 |
| `.jinja` | 2 |
| `.safetensors` | 2 |
| `.npy` | 1 |
| `[none]` | 1 |

## Largest Files

| Size MB | File |
| ---: | --- |
| 162.0 | `experiments/qwen35_4b_meta_induction/runs/lora_shift8k/adapter_model.safetensors` |
| 162.0 | `experiments/qwen35_4b_meta_induction/runs/lora_shift/adapter_model.safetensors` |
| 19.1 | `experiments/qwen35_4b_meta_induction/runs/lora_shift8k/tokenizer.json` |
| 19.1 | `experiments/qwen35_4b_meta_induction/runs/lora_shift/tokenizer.json` |
| 14.1 | `experiments/qwen35_4b_active_counterexample_trace_selection/reports/policy_rows.json` |
| 10.4 | `experiments/qwen35_4b_reliability_exec_opsd_audit/data/exec_token_pressure_scores.jsonl` |
| 8.5 | `experiments/qwen35_4b_learned_active_trace_policy/data/static_bridge_80/dsl_train.jsonl` |
| 8.4 | `experiments/qwen35_4b_foofah_program_ensemble_consensus/reports/full_ensemble_records.jsonl` |
| 8.2 | `experiments/qwen35_4b_learned_active_trace_policy/data/static_bridge_60/dsl_train.jsonl` |
| 8.0 | `experiments/qwen35_4b_learned_active_trace_policy/data/policy/policy_train.jsonl` |
| 8.0 | `experiments/qwen35_4b_inventory_shortlister_training/data/train_slots.jsonl` |
| 7.5 | `experiments/qwen35_4b_learned_active_trace_policy/data/seed/dsl_train.jsonl` |
| 7.4 | `experiments/qwen35_4b_active_counterexample_trace_selection/reports/eval/active_ceiling.json` |
| 7.3 | `experiments/qwen35_4b_learned_active_trace_policy/reports/policy_rows.json` |
| 7.0 | `experiments/qwen35_4b_learned_active_trace_policy/data/eval/dsl_eval_ceiling.jsonl` |
| 7.0 | `experiments/qwen35_4b_active_counterexample_trace_selection/data/eval/dsl_eval_ceiling.jsonl` |
| 6.4 | `experiments/qwen35_4b_active_counterexample_trace_selection/reports/eval/active_support.json` |
| 6.0 | `experiments/feature_factorized_rule_diversity/data/repair_all.jsonl` |
| 5.8 | `experiments/qwen35_4b_active_counterexample_trace_selection/data/eval/dsl_eval_support.jsonl` |
| 5.8 | `experiments/qwen35_4b_learned_active_trace_policy/data/eval/dsl_eval_support.jsonl` |
| 5.7 | `experiments/qwen35_4b_foofah_selective_program_fallback/reports/program_probe_records.jsonl` |
| 5.7 | `experiments/qwen35_4b_foofah_selective_program_fallback/reports/final_records.jsonl` |
| 5.6 | `experiments/qwen35_4b_bucket_belief_probe_ranker/data/bucket_eval_examples.jsonl` |
| 5.6 | `experiments/qwen35_4b_learned_active_trace_policy/data/base_anchor/dsl_train_static60_base.jsonl` |
| 5.4 | `experiments/qwen35_4b_foofah_selective_program_fallback/reports/selective_records.jsonl` |
| 5.0 | `experiments/qwen35_4b_learned_active_trace_policy/data/base_anchor/dsl_train_static80_base.jsonl` |
| 4.8 | `experiments/qwen35_4b_joint_shortlister_ladder/data/train_pairs.jsonl` |
| 4.7 | `experiments/qwen35_4b_joint_shortlister_ladder/data/train_records.jsonl` |
| 4.7 | `experiments/rule_family_diversity_scaling/data/repair_all.jsonl` |
| 4.6 | `experiments/qwen35_4b_oracle_probe_synthesis_mdp/reports/all_eval_records.json` |
