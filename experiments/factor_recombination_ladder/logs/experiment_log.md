# Experiment Log

## 2026-06-21

- Created standalone experiment directory at `experiments/factor_recombination_ladder/`.
- Created large-artifact directory at `large_artifacts/factor_recombination_ladder/models/`.
- Defined the central question: can trace-conditioned repair learn reusable factor recombination when specific factor-pair cells are held out from training?
- Planned a factor-balanced ladder:
  - 12 seen training rule families, 20 records each, 240 total records per training condition.
  - 5 recombination holdout families, each withholding one factor pair absent from training.
  - Three evaluation splits: seen-combination IID, format-shifted seen combinations, and recombination holdout.
  - Trace, no-trace, shuffled-trace, and factor-labelled trace training conditions.
- Implemented `scripts/build_ladder_dataset.py`.
  - The builder writes normal and factor-labelled train/eval JSONL files.
  - Each generated record is validated before writing:
    - the wrong implementation fails visible tests,
    - the target corrective diff applies,
    - the target implementation passes visible and hidden tests,
    - hidden inputs are disjoint from visible inputs.

Next step: build and validate the dataset.

### Dataset Build

- Ran `scripts/build_ladder_dataset.py` with seed `20260621`.
- The first two build attempts caught and fixed generator issues before any dataset was accepted:
  - `sum_threshold` could reuse `[1, 1]` across visible and hidden cases for one parameter draw.
  - `modulo_shift` could reuse a scalar hidden input that also appeared in visible cases.
- Both templates were patched so hidden inputs are selected disjointly from visible inputs.
- Completed dataset generation and validation.
- Wrote 1,020 total JSONL records across normal and labelled variants:
  - `repair_train_ladder`: 240 records, 12 seen rule families x 20.
  - `repair_train_ladder_labelled`: 240 records, same examples with factor labels prepended to the failing trace.
  - `repair_val_seen_iid`: 36 records, 12 seen rule families x 3.
  - `repair_val_format_shift`: 36 records, 12 seen rule families x 3.
  - `repair_val_recombination_holdout`: 60 records, 5 held-out recombination families x 12.
  - labelled variants for each evaluation split.
- Confirmed held-out factor pairs are absent from training:
  - `aggregation+modulo`
  - `branching+tuple_access`
  - `length+string_match`
  - `ordering+string_match`
  - `ordering+tuple_access`
- Manifest reports `leaked_heldout_pairs: []`.
- Compact experiment directory size after dataset build: about 5.4 MB.
- `large_artifacts/factor_recombination_ladder/` remains empty before training.

Next step: train four LoRA adapters: trace ladder, no-trace ladder, shuffled-trace ladder, and factor-labelled trace ladder.

### Training

- Ran `scripts/run_training.py --suite all`.
- Trained four LoRA adapters from `Qwen/Qwen2.5-Coder-3B-Instruct` revision `488639f1ff808d1d3d0ba301aef8c11461451ec5`.
- Shared hyperparameters:
  - max sequence length: 3072
  - epochs: 3
  - learning rate: 1.5e-4
  - LoRA rank/alpha/dropout: 32/64/0.05
  - gradient accumulation: 8
  - eval/save interval: 30 steps
- Adapters trained:
  - `ladder_trace_lora`: normal traces.
  - `ladder_no_trace_lora`: no trace in prompt.
  - `ladder_shuffled_trace_lora`: traces shuffled by seed `9173`.
  - `labelled_trace_lora`: normal traces with factor labels prepended.
- Stored all adapters and checkpoints outside the compact experiment directory under `large_artifacts/factor_recombination_ladder/models/`.
- Training console log: `run_logs/training_console.log`.
- Training manifest: `reports/training/training_jobs.json`.
- End-of-training loss pattern:
  - trace and labelled-trace adapters learned much lower training loss than no-trace and shuffled-trace controls.
  - held-out validation loss remained substantially higher than seen-format validation for all conditions.

Next step: run the full 30-job final evaluation matrix.

### Final Evaluation

- Ran `scripts/run_final_evaluations.py --suite all`.
- Completed all 30 planned final evaluation jobs:
  - 5 core conditions x 3 splits.
  - 5 prompt ablation conditions x 3 splits.
- Final evaluation console log: `run_logs/final_evaluation_console.log`.
- Final evaluation manifest: `reports/final/final_evaluation_jobs.json`.
- Machine-readable summaries:
  - `reports/final_results.csv`
  - `reports/final_results_by_family.csv`
  - `reports/final_results_by_factor.csv`

Core repair@1 results:

| Condition | Seen-IID | Format shift | Recombination holdout |
| --- | ---: | ---: | ---: |
| Frozen trace | 0.0% (0/36) | 0.0% (0/36) | 0.0% (0/60) |
| Trace ladder | 80.6% (29/36) | 63.9% (23/36) | 8.3% (5/60) |
| No-trace ladder | 33.3% (12/36) | 25.0% (9/36) | 10.0% (6/60) |
| Shuffled-trace ladder | 27.8% (10/36) | 30.6% (11/36) | 8.3% (5/60) |
| Factor-labelled trace ladder | 80.6% (29/36) | 72.2% (26/36) | 8.3% (5/60) |

Prompt ablation results:

| Condition | Seen-IID | Format shift | Recombination holdout |
| --- | ---: | ---: | ---: |
| Trace ladder, no trace prompt | 16.7% (6/36) | 19.4% (7/36) | 8.3% (5/60) |
| Trace ladder, shuffled trace prompt | 0.0% (0/36) | 2.8% (1/36) | 6.7% (4/60) |
| Labelled adapter, labels removed | 75.0% (27/36) | 69.4% (25/36) | 5.0% (3/60) |
| Labelled adapter, no trace prompt | 22.2% (8/36) | 16.7% (6/36) | 8.3% (5/60) |
| Labelled adapter, shuffled trace prompt | 13.9% (5/36) | 8.3% (3/36) | 6.7% (4/60) |

Recombination holdout by family showed the small number of successes concentrated in `sorted_tuple_affine`; three of five held-out families were at or near zero across all trained conditions.

### Interpretation

- Correct trace supervision has a large effect on seen-combination repair and format-shift repair.
- Shuffled traces and no-trace prompts substantially degrade seen and format-shift performance, so the trace content is behaviorally important.
- Factor labels help format shift, and most of that benefit remains when labels are removed at inference time, suggesting labels improve training organization more than acting as a required runtime token.
- The key negative result is stable: trace supervision did not produce robust transfer to unseen factor-pair recombinations in this setup.
- Best core recombination score was the no-trace ladder at 10.0% (6/60), only one example above trace and labelled-trace at 8.3% (5/60).
- This experiment should be treated as evidence that the next most useful direction is not more trace decoration, but an intervention that directly trains or searches over factor recombination.

### Wrap-Up

- Generated final markdown report: `reports/factor_recombination_ladder_report.md`.
- Generated figures:
  - `figures/final_repair_by_condition_split.png`
  - `figures/recombination_holdout_by_family.png`
- Verified artifact split:
  - compact experiment directory: about 14 MB.
  - large artifact directory: about 5.2 GB.
  - no `.safetensors`, `.bin`, `.pt`, or `.pth` files were present in `experiments/factor_recombination_ladder/`.
- Final status: complete.
