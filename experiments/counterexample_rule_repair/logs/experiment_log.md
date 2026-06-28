# Counterexample Rule Repair Experiment Log

## 2026-06-20 Setup

Objective: design and run a standalone counterexample-to-rule repair experiment where the failed execution trace contains concrete examples of the desired behavior, and hidden tests require applying the inferred rule to new inputs.

Directory policy:

- Small, downloadable experiment package: `/workspace/experiments/counterexample_rule_repair/`.
- Large artifacts excluded from the small package: `/workspace/large_artifacts/counterexample_rule_repair/`.
- Model adapters and checkpoints go under `large_artifacts/counterexample_rule_repair/models/`.
- Reports, logs, configs, figures, and compact JSON/JSONL result summaries go under `experiments/counterexample_rule_repair/`.

Initial hypothesis:

A repair model trained on `wrong patched state + failed execution counterexamples -> corrective diff` will outperform no-trace and shuffled-trace controls when the correct fix requires inferring a compact rule from visible counterexamples and then passing hidden tests on unseen inputs.

Design constraints:

- The paper and artifacts must be standalone.
- The expected rule parameters must not appear in the issue text.
- The visible trace must contain enough counterexamples to infer the rule.
- Hidden tests must require generalization beyond the visible counterexamples.

## 2026-06-20 Dataset Design

Primary task shape:

- Each record contains one production file, `src/repair_target.py`, with an `apply_rule(value)` function.
- The issue text says the validator rule is not stated and must be inferred from failed-test counterexamples.
- The wrong-patched implementation is a compact but incorrect rule.
- The failed visible test emits `COUNTEREXAMPLE input=... expected=... actual=...` lines.
- The target corrective diff patches the implementation into a compact rule.
- Hidden tests use inputs that are not present in the visible trace, so a patch that hardcodes only visible counterexamples should fail.

Train and IID/format-holdout families:

- `affine_int`: infer a linear integer rule from numeric counterexamples.
- `threshold_label`: infer a threshold and two output labels from boundary counterexamples.
- `slug_affix`: infer string affixes and separator while preserving slug normalization behavior.

Rule-family holdout:

- `parity_offset_holdout`: infer separate offsets for even and odd inputs. This family is withheld from training.

Validation performed by the builder:

- The wrong patch fails visible tests.
- The target corrective diff applies to the wrong-patched implementation.
- The repaired implementation passes visible and hidden tests.
- The repaired implementation compiles.
- Hidden test inputs do not overlap visible trace inputs.
- Visible expected outputs appear in the failed execution trace.

Smoke build:

- A 14-record smoke dataset passed all builder invariants.
- Inspected affine and threshold examples confirmed that visible counterexamples are present in trace output and hidden examples require unseen inputs.

## 2026-06-20 Full Dataset Build

Command:

`python experiments/counterexample_rule_repair/scripts/build_counterexample_dataset.py --output-dir experiments/counterexample_rule_repair/data --train-per-family 80 --iid-per-family 15 --format-per-family 15 --rule-holdout 45 --seed 20260620`

Result:

- Train: 240 records.
- IID validation: 45 records.
- Format-holdout validation: 45 records.
- Rule-family-holdout validation: 45 records.
- All records: 375.
- Data directory size: about 4.3 MB.

Validation split composition:

- IID: 15 `affine_int`, 15 `threshold_label`, 15 `slug_affix`.
- Format holdout: 15 `affine_int`, 15 `threshold_label`, 15 `slug_affix`, with shifted numeric ranges and different string-token formats.
- Rule-family holdout: 45 `parity_offset_holdout` records.

All builder invariants passed.

## 2026-06-20 Frozen Pilot

Command:

`python experiments/counterexample_rule_repair/scripts/eval_counterexample_rule.py --data experiments/counterexample_rule_repair/data/repair_val_iid.jsonl --output experiments/counterexample_rule_repair/reports/frozen_trace_iid_pilot6.json --condition trace --max-records 6 --max-new-tokens 256`

Result:

- Model: `Qwen/Qwen2.5-Coder-3B-Instruct`, revision `488639f1ff808d1d3d0ba301aef8c11461451ec5`.
- Records: 6 IID validation examples.
- Repair@1: 0/6.
- Visible pass rate: 0/6.
- Patch apply rate: 1/6.
- Target-added-line match rate: 0/6.

Observed failure mode:

- The frozen model often generated diffs against the original placeholder implementation rather than the wrong-patched state.
- When a patch applied, it preserved wrong constants rather than deriving the target rule from counterexamples.

Next step:

- Train a small trace-conditioned pilot adapter on 90 training records for one epoch, then evaluate trace, no-trace, and shuffled-trace behavior.

## 2026-06-20 Trace Pilot Training and Controls

Training command:

`python scripts/train_repair_lora.py --train experiments/counterexample_rule_repair/data/repair_train.jsonl --eval experiments/counterexample_rule_repair/data/repair_val_iid.jsonl --mode trace --model-id Qwen/Qwen2.5-Coder-3B-Instruct --revision 488639f1ff808d1d3d0ba301aef8c11461451ec5 --output-dir large_artifacts/counterexample_rule_repair/models/pilot_trace_lora --max-length 3072 --epochs 1 --lr 2e-4 --rank 16 --alpha 32 --dropout 0.05 --grad-accum 8 --save-steps 20 --eval-steps 20 --max-train-records 90`

Training observations:

- Trainable parameters: 29,933,568, about 0.96% of the model.
- Training took about 57 seconds.
- Train loss: `0.09427`.
- Eval loss on all 45 IID validation records: `0.03152`.

Evaluation metric correction:

- Initial pilot inspection showed that hidden-only pass can overstate success when a patch passes unseen hidden inputs but fails visible counterexamples.
- Updated evaluator so `repair@1` requires both visible and hidden tests to pass.
- Added `hidden_pass_rate` as a separate diagnostic.

Pilot evaluations, all on the first 20 IID validation records:

- Frozen base + trace: 0/6 in the earlier smoke pilot.
- Pilot trace adapter + trace: 9/20 repair@1, 20/20 patch apply, 20/20 syntax valid.
- Pilot trace adapter + no trace: 0/20 repair@1, 20/20 patch apply, 20/20 syntax valid.
- Pilot trace adapter + shuffled trace: 0/20 repair@1, 20/20 patch apply, 20/20 syntax valid.

Per-family trace pilot notes:

- `slug_affix`: 4/5 repair@1.
- `threshold_label`: 4/8 repair@1; several failures were off-by-one thresholds that passed hidden examples but failed visible counterexamples.
- `affine_int`: 1/7 repair@1; arithmetic inference from counterexamples is the hardest family.

Decision:

- The pilot establishes a causal trace contrast but is not saturated.
- Proceed to a stronger full trace adapter first: all 240 train records, 3 epochs, rank 32, alpha 64, max length 3072.
- Train the full control adapters after confirming the stronger trace adapter improves.

## 2026-06-20 Full Trace Adapter

Training command:

`python scripts/train_repair_lora.py --train experiments/counterexample_rule_repair/data/repair_train.jsonl --eval experiments/counterexample_rule_repair/data/repair_val_iid.jsonl --mode trace --model-id Qwen/Qwen2.5-Coder-3B-Instruct --revision 488639f1ff808d1d3d0ba301aef8c11461451ec5 --output-dir large_artifacts/counterexample_rule_repair/models/trace_lora --max-length 3072 --epochs 3 --lr 1.5e-4 --rank 32 --alpha 64 --dropout 0.05 --grad-accum 8 --save-steps 30 --eval-steps 30`

Training observations:

- Trainable parameters: 59,867,136, about 1.90% of the model.
- Training took about 422 seconds.
- Final train loss: `0.01636`.
- Final IID eval loss: `0.004355`.

Sanity evaluation command:

`python experiments/counterexample_rule_repair/scripts/eval_counterexample_rule.py --data experiments/counterexample_rule_repair/data/repair_val_iid.jsonl --output experiments/counterexample_rule_repair/reports/full_trace_iid20_check.json --condition trace --adapter large_artifacts/counterexample_rule_repair/models/trace_lora --max-records 20 --max-new-tokens 256`

Result:

- Records: 20 IID validation examples.
- Repair@1: 17/20.
- Visible pass rate: 17/20.
- Hidden pass rate: 17/20.
- Patch apply rate: 20/20.
- Syntax valid rate: 20/20.

Per-family sanity notes:

- `slug_affix`: 5/5 repair@1.
- `threshold_label`: 8/8 repair@1.
- `affine_int`: 4/7 repair@1.

Decision:

- The full trace adapter is strong enough to justify the full control suite.
- Use the same model, revision, LoRA rank, alpha, dropout, epochs, learning rate, and max length for the no-trace, shuffled-trace, and final-patch controls.

## 2026-06-20 Full Control Adapter Training

No-trace training command:

`python scripts/train_repair_lora.py --train experiments/counterexample_rule_repair/data/repair_train.jsonl --eval experiments/counterexample_rule_repair/data/repair_val_iid.jsonl --mode no_trace --model-id Qwen/Qwen2.5-Coder-3B-Instruct --revision 488639f1ff808d1d3d0ba301aef8c11461451ec5 --output-dir large_artifacts/counterexample_rule_repair/models/no_trace_lora --max-length 3072 --epochs 3 --lr 1.5e-4 --rank 32 --alpha 64 --dropout 0.05 --grad-accum 8 --save-steps 30 --eval-steps 30`

No-trace result:

- Trainable parameters: 59,867,136, about 1.90% of the model.
- Training took about 422 seconds.
- Final train loss: `0.2545`.
- Final IID eval loss: `0.2151`.

Shuffled-trace training command:

`python scripts/train_repair_lora.py --train experiments/counterexample_rule_repair/data/repair_train.jsonl --eval experiments/counterexample_rule_repair/data/repair_val_iid.jsonl --mode trace --shuffle-traces --model-id Qwen/Qwen2.5-Coder-3B-Instruct --revision 488639f1ff808d1d3d0ba301aef8c11461451ec5 --output-dir large_artifacts/counterexample_rule_repair/models/shuffled_trace_lora --max-length 3072 --epochs 3 --lr 1.5e-4 --rank 32 --alpha 64 --dropout 0.05 --grad-accum 8 --save-steps 30 --eval-steps 30`

Shuffled-trace result:

- Trainable parameters: 59,867,136, about 1.90% of the model.
- Training took about 417 seconds.
- Final train loss: `0.2538`.
- Final IID eval loss: `0.2106`.

Final-patch training command:

`python scripts/train_repair_lora.py --train experiments/counterexample_rule_repair/data/repair_train.jsonl --eval experiments/counterexample_rule_repair/data/repair_val_iid.jsonl --mode final_patch --model-id Qwen/Qwen2.5-Coder-3B-Instruct --revision 488639f1ff808d1d3d0ba301aef8c11461451ec5 --output-dir large_artifacts/counterexample_rule_repair/models/final_patch_lora --max-length 3072 --epochs 3 --lr 1.5e-4 --rank 32 --alpha 64 --dropout 0.05 --grad-accum 8 --save-steps 30 --eval-steps 30`

Final-patch result:

- Trainable parameters: 59,867,136, about 1.90% of the model.
- Training took about 426 seconds.
- Final train loss: `0.3666`.
- Final IID eval loss: `0.2827`.

Interpretation before final evaluations:

- The trace adapter fit the supervised target much more closely than the no-trace, shuffled-trace, and final-patch controls.
- This does not prove repair success by itself, so the final comparison must use generated patches run against visible and hidden tests.

## 2026-06-20 Final Evaluation Plan

Evaluation runner:

`python experiments/counterexample_rule_repair/scripts/run_final_evaluations.py --suite all --max-new-tokens 256`

Planned splits:

- IID validation: 45 records from the training rule families.
- Format holdout: 45 records from the training rule families with shifted numeric ranges and token formats.
- Rule-family holdout: 45 `parity_offset_holdout` records, with the family absent from training.

Core conditions:

- Frozen base + trace.
- Final-patch SFT + final patch.
- No-trace SFT + no trace.
- Shuffled-trace SFT + trace.
- Trace SFT + trace.

Trace-adapter input ablations:

- Trace SFT + no trace.
- Trace SFT + shuffled trace.

## 2026-06-20 Final Evaluation Results

Final evaluation command:

`python experiments/counterexample_rule_repair/scripts/run_final_evaluations.py --suite all --max-new-tokens 256`

Result files:

- 21 final JSON result files were written to `experiments/counterexample_rule_repair/reports/final_*.json`.
- Each final JSON contains aggregate metrics plus per-record completions, extracted patches, patch-application status, syntax status, visible-test output, and hidden-test output.

Core results:

| Split | Frozen + trace | Final-patch SFT | No-trace SFT | Shuffled-trace SFT | Trace SFT + trace |
| --- | ---: | ---: | ---: | ---: | ---: |
| IID | 0/45 | 0/45 | 4/45 | 2/45 | 41/45 |
| Format holdout | 0/45 | 0/45 | 0/45 | 0/45 | 24/45 |
| Rule-family holdout | 0/45 | 0/45 | 1/45 | 0/45 | 0/45 |

Trace adapter input ablations:

| Split | Trace SFT + no trace | Trace SFT + shuffled trace |
| --- | ---: | ---: |
| IID | 0/45 | 0/45 |
| Format holdout | 0/45 | 0/45 |
| Rule-family holdout | 0/45 | 0/45 |

Family breakdown for Trace SFT + trace:

- IID: `affine_int` 11/15, `slug_affix` 15/15, `threshold_label` 15/15.
- Format holdout: `affine_int` 0/15, `slug_affix` 9/15, `threshold_label` 15/15.
- Rule-family holdout: `parity_offset_holdout` 0/45.

Interpretation:

- The aligned trace condition is the only condition with strong IID and format-holdout repair.
- Valid diff generation alone is not enough: final-patch, no-trace, shuffled-trace, and input-ablation conditions often reached 100% patch application while still failing visible and hidden tests.
- The trace adapter remained dependent on aligned trace evidence at inference time: removing or shuffling the trace reduced repair@1 to 0/45 on every split.
- The withheld `parity_offset_holdout` family did not transfer. The trace adapter generated applicable, syntactically valid diffs, but failed every visible and hidden test in that split.

## 2026-06-20 Report Generation

Report command:

`python experiments/counterexample_rule_repair/scripts/make_report.py`

Generated small-package report artifacts:

- `reports/counterexample_rule_repair_paper.md`
- `reports/counterexample_rule_repair_summary.md`
- `reports/final_core_results.csv`
- `reports/final_ablation_results.csv`
- `reports/final_trace_by_family.csv`
- `reports/pilot_results.csv`
- `figures/core_repair_rates.png`
- `figures/trace_ablation_repair_rates.png`
- `figures/visible_pass_rates.png`
- `large_artifacts_manifest.md`

Artifact split:

- Small, download-friendly package: `experiments/counterexample_rule_repair/`.
- Large adapters/checkpoints: `large_artifacts/counterexample_rule_repair/models/`.
- The small package intentionally contains no model adapter files.

## 2026-06-20 Final Verification

Verification commands and outcomes:

- Script compilation passed:
  `python -m py_compile experiments/counterexample_rule_repair/scripts/build_counterexample_dataset.py experiments/counterexample_rule_repair/scripts/eval_counterexample_rule.py experiments/counterexample_rule_repair/scripts/run_final_evaluations.py experiments/counterexample_rule_repair/scripts/make_report.py scripts/train_repair_lora.py`
- Final result count: 21 `reports/final_*.json` files.
- Figure files are nonempty:
  - `figures/core_repair_rates.png`
  - `figures/trace_ablation_repair_rates.png`
  - `figures/visible_pass_rates.png`
- Small-package size: about 9.9 MB.
- Large-artifact directory size: about 5.5 GB.
- No files larger than 50 MB were found under `experiments/counterexample_rule_repair/`.
- No `.safetensors`, `.bin`, `.pt`, or `.pth` model-weight files were found under `experiments/counterexample_rule_repair/`.
- No active `train_repair_lora`, `run_final_evaluations`, or `eval_counterexample_rule` processes remained after the final sweep.
- Removed generated `__pycache__` and `.ipynb_checkpoints` directories from the small package.
- Reference hygiene check found no stale references to unrelated experiment names in the counterexample-rule package.
