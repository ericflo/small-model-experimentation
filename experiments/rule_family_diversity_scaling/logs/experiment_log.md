# Rule-Family Diversity Scaling Experiment Log

## 2026-06-21 Setup

Objective: run a standalone rule-family diversity scaling experiment that tests whether trace-conditioned repair learns a transferable counterexample-to-rule procedure as training family diversity increases.

Directory policy:

- Small, downloadable experiment package: `/workspace/experiments/rule_family_diversity_scaling/`.
- Large artifacts excluded from the small package: `/workspace/large_artifacts/rule_family_diversity_scaling/`.
- Model adapters and checkpoints go under `large_artifacts/rule_family_diversity_scaling/models/`.
- Reports, logs, configs, figures, and compact JSON/JSONL result summaries go under `experiments/rule_family_diversity_scaling/`.

Initial hypothesis:

If trace-conditioned repair is learning a general counterexample-to-rule procedure, then adapters trained on more diverse rule families should improve on held-out rule families even when total training record count is held fixed.

Design constraints:

- The paper and artifacts must be standalone.
- Total training records are fixed at 240 for every diversity scale.
- The exact target rule is not stated in issue text.
- Visible failure traces contain concrete counterexamples.
- Hidden tests use inputs not present in the visible trace.
- Final evaluation must separately report trained-family IID, trained-family format holdout, and fully held-out rule-family transfer.

## 2026-06-21 Dataset Builder Smoke Test

Builder:

`experiments/rule_family_diversity_scaling/scripts/build_diversity_dataset.py`

Smoke command:

`python experiments/rule_family_diversity_scaling/scripts/build_diversity_dataset.py --output-dir /tmp/rfds_smoke --total-train-records 12 --base-iid-per-family 1 --format-per-family 1 --holdout-per-family 1 --seed 20260621`

Smoke result:

- Build completed successfully.
- Each diversity scale had 12 training records.
- Validation smoke splits had 3 base-IID records, 3 format-holdout records, and 4 held-out-family records.
- The builder initially exposed a useful edge case: numeric wrong rules can coincidentally pass one visible case, which would omit that expected output from the failure trace. The builder now rejects and resamples any record that does not satisfy the trace-evidence invariant.

## 2026-06-21 Full Dataset Build

Command:

`python experiments/rule_family_diversity_scaling/scripts/build_diversity_dataset.py --output-dir experiments/rule_family_diversity_scaling/data --total-train-records 240 --base-iid-per-family 12 --format-per-family 12 --holdout-per-family 12 --seed 20260621`

Result:

- `train_scale3`: 240 records from 3 families, 80 records per family.
- `train_scale6`: 240 records from 6 families, 40 records per family.
- `train_scale12`: 240 records from 12 families, 20 records per family.
- `val_base_iid`: 36 records from the 3 base families.
- `val_format_holdout`: 36 records from the 3 base families with shifted numeric ranges and token formats.
- `val_rule_holdout`: 48 records from 4 families absent from every training scale.
- `repair_all`: 840 total records.

Training-family scales:

- `scale3`: `affine_int`, `threshold_label`, `slug_affix`.
- `scale6`: `affine_int`, `threshold_label`, `slug_affix`, `abs_shift`, `clamp_offset`, `tuple_linear`.
- `scale12`: `affine_int`, `threshold_label`, `slug_affix`, `abs_shift`, `clamp_offset`, `tuple_linear`, `length_label`, `contains_label`, `prefix_switch`, `modulo_label`, `sign_piece`, `replace_wrap`.

Held-out rule families:

- `parity_offset_holdout`
- `quadratic_shift_holdout`
- `tuple_max_holdout`
- `sorted_join_holdout`

All builder invariants passed.

## 2026-06-21 Frozen Evaluation Smoke

Command:

`python experiments/rule_family_diversity_scaling/scripts/eval_diversity.py --data experiments/rule_family_diversity_scaling/data/repair_val_base_iid.jsonl --output experiments/rule_family_diversity_scaling/reports/frozen_trace_base_iid_pilot6.json --condition trace --max-records 6 --max-new-tokens 256`

Result:

- Records: 6 base-IID examples.
- Repair@1: 0/6.
- Patch apply rate: 0/6.
- Visible pass rate: 0/6.

Decision:

- The evaluator works on the new record schema.
- Proceed with LoRA training for diversity scales and controls.

## 2026-06-21 Training: scale3_trace_lora

Command:

`python scripts/train_repair_lora.py --train experiments/rule_family_diversity_scaling/data/repair_train_scale3.jsonl --eval experiments/rule_family_diversity_scaling/data/repair_val_base_iid.jsonl --mode trace --model-id Qwen/Qwen2.5-Coder-3B-Instruct --revision 488639f1ff808d1d3d0ba301aef8c11461451ec5 --output-dir large_artifacts/rule_family_diversity_scaling/models/scale3_trace_lora --max-length 3072 --epochs 3 --lr 1.5e-4 --rank 32 --alpha 64 --dropout 0.05 --grad-accum 8 --save-steps 30 --eval-steps 30`

Result:

- Completed successfully.
- Output directory: `large_artifacts/rule_family_diversity_scaling/models/scale3_trace_lora`.
- Trainable parameters: 59,867,136 (1.9031%).
- Training steps: 90.
- Train runtime: 411 seconds.
- Final train loss: 0.01668.
- Evaluation loss by epoch on `val_base_iid`: epoch 1 = 0.004631, epoch 2 = 0.003665, epoch 3 = 0.001845.
- Checkpoints and adapter weights were written under `large_artifacts/`, not under the downloadable experiment package.

## 2026-06-21 Training: scale6_trace_lora

Command:

`python scripts/train_repair_lora.py --train experiments/rule_family_diversity_scaling/data/repair_train_scale6.jsonl --eval experiments/rule_family_diversity_scaling/data/repair_val_base_iid.jsonl --mode trace --model-id Qwen/Qwen2.5-Coder-3B-Instruct --revision 488639f1ff808d1d3d0ba301aef8c11461451ec5 --output-dir large_artifacts/rule_family_diversity_scaling/models/scale6_trace_lora --max-length 3072 --epochs 3 --lr 1.5e-4 --rank 32 --alpha 64 --dropout 0.05 --grad-accum 8 --save-steps 30 --eval-steps 30`

Result:

- Completed successfully.
- Output directory: `large_artifacts/rule_family_diversity_scaling/models/scale6_trace_lora`.
- Trainable parameters: 59,867,136 (1.9031%).
- Training steps: 90.
- Train runtime: 415.4 seconds.
- Final train loss: 0.02493.
- Evaluation loss by epoch on `val_base_iid`: epoch 1 = 0.01205, epoch 2 = 0.01092, epoch 3 = 0.0101.
- Checkpoints and adapter weights were written under `large_artifacts/`, not under the downloadable experiment package.

## 2026-06-21 Training: scale12_trace_lora

Command:

`python scripts/train_repair_lora.py --train experiments/rule_family_diversity_scaling/data/repair_train_scale12.jsonl --eval experiments/rule_family_diversity_scaling/data/repair_val_base_iid.jsonl --mode trace --model-id Qwen/Qwen2.5-Coder-3B-Instruct --revision 488639f1ff808d1d3d0ba301aef8c11461451ec5 --output-dir large_artifacts/rule_family_diversity_scaling/models/scale12_trace_lora --max-length 3072 --epochs 3 --lr 1.5e-4 --rank 32 --alpha 64 --dropout 0.05 --grad-accum 8 --save-steps 30 --eval-steps 30`

Result:

- Completed successfully.
- Output directory: `large_artifacts/rule_family_diversity_scaling/models/scale12_trace_lora`.
- Trainable parameters: 59,867,136 (1.9031%).
- Training steps: 90.
- Train runtime: 393.7 seconds.
- Final train loss: 0.02588.
- Evaluation loss by epoch on `val_base_iid`: epoch 1 = 0.0199, epoch 2 = 0.01371, epoch 3 = 0.01347.
- Checkpoints and adapter weights were written under `large_artifacts/`, not under the downloadable experiment package.

## 2026-06-21 Training: scale12_no_trace_lora

Command:

`python scripts/train_repair_lora.py --train experiments/rule_family_diversity_scaling/data/repair_train_scale12.jsonl --eval experiments/rule_family_diversity_scaling/data/repair_val_base_iid.jsonl --mode no_trace --model-id Qwen/Qwen2.5-Coder-3B-Instruct --revision 488639f1ff808d1d3d0ba301aef8c11461451ec5 --output-dir large_artifacts/rule_family_diversity_scaling/models/scale12_no_trace_lora --max-length 3072 --epochs 3 --lr 1.5e-4 --rank 32 --alpha 64 --dropout 0.05 --grad-accum 8 --save-steps 30 --eval-steps 30`

Result:

- Completed successfully.
- Output directory: `large_artifacts/rule_family_diversity_scaling/models/scale12_no_trace_lora`.
- Trainable parameters: 59,867,136 (1.9031%).
- Training steps: 90.
- Train runtime: 391 seconds.
- Final train loss: 0.2446.
- Evaluation loss by epoch on `val_base_iid`: epoch 1 = 0.2427, epoch 2 = 0.2351, epoch 3 = 0.2393.
- Checkpoints and adapter weights were written under `large_artifacts/`, not under the downloadable experiment package.

## 2026-06-21 Training: scale12_shuffled_trace_lora

Command:

`python scripts/train_repair_lora.py --train experiments/rule_family_diversity_scaling/data/repair_train_scale12.jsonl --eval experiments/rule_family_diversity_scaling/data/repair_val_base_iid.jsonl --mode trace --shuffle-traces --model-id Qwen/Qwen2.5-Coder-3B-Instruct --revision 488639f1ff808d1d3d0ba301aef8c11461451ec5 --output-dir large_artifacts/rule_family_diversity_scaling/models/scale12_shuffled_trace_lora --max-length 3072 --epochs 3 --lr 1.5e-4 --rank 32 --alpha 64 --dropout 0.05 --grad-accum 8 --save-steps 30 --eval-steps 30`

Result:

- Completed successfully.
- Output directory: `large_artifacts/rule_family_diversity_scaling/models/scale12_shuffled_trace_lora`.
- Trainable parameters: 59,867,136 (1.9031%).
- Training steps: 90.
- Train runtime: 394.4 seconds.
- Final train loss: 0.2452.
- Evaluation loss by epoch on `val_base_iid`: epoch 1 = 0.2418, epoch 2 = 0.2307, epoch 3 = 0.2331.
- Checkpoints and adapter weights were written under `large_artifacts/`, not under the downloadable experiment package.

## 2026-06-21 Final Evaluation Suite

Command:

`python experiments/rule_family_diversity_scaling/scripts/run_final_evaluations.py --force`

Evaluation protocol:

- Deterministic generation with `max_new_tokens=256`.
- `repair@1` requires both visible and hidden tests to pass.
- Full splits were evaluated: 36 base-IID records, 36 format-holdout records, and 48 held-out rule-family records.
- Final outputs were written to `experiments/rule_family_diversity_scaling/reports/final/`.
- The final suite produced 18 core result JSONs, 6 ablation result JSONs, and `final_evaluation_jobs.json`.

Core repair@1 results:

| Condition | Base IID | Format Holdout | Rule Holdout |
| --- | ---: | ---: | ---: |
| Frozen trace | 0/36 | 0/36 | 0/48 |
| scale3 trace | 34/36 | 18/36 | 0/48 |
| scale6 trace | 28/36 | 19/36 | 2/48 |
| scale12 trace | 31/36 | 16/36 | 14/48 |
| scale12 no-trace train/eval | 0/36 | 0/36 | 1/48 |
| scale12 shuffled-trace train, real trace eval | 0/36 | 0/36 | 1/48 |

Scale12 trace adapter prompt ablations:

| Prompt condition | Base IID | Format Holdout | Rule Holdout |
| --- | ---: | ---: | ---: |
| Real trace | 31/36 | 16/36 | 14/48 |
| No trace | 0/36 | 0/36 | 0/48 |
| Shuffled trace | 0/36 | 0/36 | 2/48 |

Held-out family detail for `scale12_trace`:

- `parity_offset_holdout`: 0/12.
- `quadratic_shift_holdout`: 3/12.
- `sorted_join_holdout`: 11/12.
- `tuple_max_holdout`: 0/12.

Interpretation:

- Increasing trace-training diversity from 3 to 12 rule families improved held-out rule-family repair from 0/48 to 14/48 under the same 240-record training budget.
- The transfer gain was not uniform across held-out families; it was concentrated in `sorted_join_holdout` and, to a smaller extent, `quadratic_shift_holdout`.
- Valid trace evidence was necessary for the observed transfer: no-trace training, shuffled-trace training, no-trace prompting, and shuffled-trace prompting all scored at or near zero on the same held-out split.
- The broadest diversity condition did not dominate every split: `scale3_trace` was best on base-IID and `scale6_trace` was best on format holdout.

## 2026-06-21 Report Generation

Command:

`python experiments/rule_family_diversity_scaling/scripts/make_report.py`

Generated compact artifacts:

- `reports/rule_family_diversity_scaling_paper.md`
- `reports/rule_family_diversity_scaling_summary.md`
- `reports/final_core_results.csv`
- `reports/final_ablation_results.csv`
- `reports/final_scale_by_split.csv`
- `reports/final_trace_by_family.csv`
- `reports/pilot_results.csv`
- `figures/final_repair_by_condition_split.png`
- `figures/diversity_scale_curve.png`
- `figures/scale12_trace_ablation.png`

Packaging checks:

- Compact experiment directory size after report generation: 15 MB.
- Large artifact directory size after training: 6.4 GB.
- No `*.safetensors`, `*.bin`, `*.pt`, or `*.pth` files were found under `experiments/rule_family_diversity_scaling/`.
- No files larger than 50 MB were found under `experiments/rule_family_diversity_scaling/`.
- Adapter weights and checkpoints are stored under `large_artifacts/rule_family_diversity_scaling/models/`.

## 2026-06-21 Final Verification

Verification commands checked:

- Final evaluation manifest: 24 jobs, 24 completed.
- Final result JSONs: 18 core JSON files and 6 ablation JSON files.
- Generated reports, CSVs, and figures are present.
- No `__pycache__` or `.ipynb_checkpoints` directories remain in the compact package.
- No active `train_repair_lora.py`, `eval_diversity.py`, or `run_final_evaluations.py` processes remain.
- No model weight/checkpoint files or files larger than 50 MB were found under `experiments/rule_family_diversity_scaling/`.
- No references to earlier experiment directory names were found under `experiments/rule_family_diversity_scaling/`.

Final package sizes:

- Compact package: `experiments/rule_family_diversity_scaling/` = 15 MB.
- Large artifacts: `large_artifacts/rule_family_diversity_scaling/` = 6.4 GB.

Conclusion:

- The experiment is complete and packaged with small artifacts separated from adapters/checkpoints.
