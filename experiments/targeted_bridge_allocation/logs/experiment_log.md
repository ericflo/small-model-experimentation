# Experiment Log

## 2026-06-22

- Created standalone experiment directory at `experiments/targeted_bridge_allocation/`.
- Created large-artifact directory at `large_artifacts/targeted_bridge_allocation/models/`.
- Defined the central question: whether targeted allocation of a fixed bridge-example budget improves trace-conditioned recombination repair compared with uniform allocation.
- Planned a fixed-budget allocation experiment:
  - every trained condition has 240 records,
  - uniform baselines allocate 2 or 4 exact bridge examples to every held-out family,
  - the main targeted condition allocates 8 bridge examples to each target family and 2 to each responsive-control family,
  - a seen-preserving targeted condition keeps the total bridge count equal to the uniform4 baseline,
  - an easy-target control spends the same bridge budget as the main targeted condition on responsive-control families,
  - three light single-family probes raise one target family at a time to 16 bridge examples,
  - no-trace and shuffled-trace controls test whether the targeted allocation depends on aligned trace evidence.
- Implemented the experiment package:
  - `scripts/build_allocation_dataset.py`
  - `scripts/eval_allocation.py`
  - `scripts/run_training.py`
  - `scripts/run_final_evaluations.py`
  - `scripts/make_report.py`
- The builder validates every generated record before writing:
  - wrong implementation fails visible tests,
  - target corrective diff applies,
  - target implementation passes visible and hidden tests,
  - visible and hidden inputs are disjoint.

Next step: build and validate the datasets, then inspect the manifest before starting training.

### Dataset Build

- Ran `scripts/build_allocation_dataset.py` with seed `20260622`.
- Completed executable validation for every generated record.
- Wrote eight 240-record training files:
  - `repair_train_uniform2.jsonl`
  - `repair_train_uniform4.jsonl`
  - `repair_train_hard_target.jsonl`
  - `repair_train_hard_target_seen_preserving.jsonl`
  - `repair_train_easy_target_control.jsonl`
  - `repair_train_modulo16.jsonl`
  - `repair_train_length16.jsonl`
  - `repair_train_tuple16.jsonl`
- Wrote three fixed evaluation splits:
  - `repair_val_seen_iid.jsonl`: 36 records.
  - `repair_val_format_shift.jsonl`: 36 records.
  - `repair_val_recombination_holdout.jsonl`: 60 records.
- Verified allocation invariants from `data/dataset_manifest.json`:
  - `uniform2`: 2 bridge examples for every held-out factor pair.
  - `uniform4`: 4 bridge examples for every held-out factor pair.
  - `hard_target`: 8 bridge examples for `aggregation+modulo`, `branching+tuple_access`, and `length+string_match`; 2 each for `ordering+string_match` and `ordering+tuple_access`.
  - `hard_target_seen_preserving`: 6 target-family bridge examples and 1 responsive-control bridge example for each responsive-control pair; total bridge count matches `uniform4`.
  - `easy_target_control`: same bridge total as `hard_target`, but 11 bridge examples each for `ordering+string_match` and `ordering+tuple_access`.
  - `modulo16`: 16 bridge examples for `aggregation+modulo`; 2 for every other held-out factor pair.
  - `length16`: 16 bridge examples for `length+string_match`; 2 for every other held-out factor pair.
  - `tuple16`: 16 bridge examples for `branching+tuple_access`; 2 for every other held-out factor pair.
- Compact experiment directory after dataset build: about 15 MB.
- Large artifact directory remains empty before training.
- Confirmed no `.safetensors`, `.bin`, `.pt`, or `.pth` files are present in the compact experiment directory.

Next step: train the allocation adapters and controls.

### Training

- Ran `scripts/run_training.py --suite all`.
- Completed 10/10 LoRA training jobs.
- Wrote adapter and checkpoint artifacts under `large_artifacts/targeted_bridge_allocation/models/`.
- Training runtimes:
  - `uniform2_trace`: 429.947 seconds.
  - `uniform4_trace`: 442.551 seconds.
  - `hard_target_trace`: 446.382 seconds.
  - `hard_target_seen_preserving_trace`: 420.899 seconds.
  - `easy_target_control_trace`: 441.100 seconds.
  - `modulo16_trace`: 428.771 seconds.
  - `length16_trace`: 444.229 seconds.
  - `tuple16_trace`: 426.457 seconds.
  - `hard_target_no_trace`: 437.514 seconds.
  - `hard_target_shuffled_trace`: 416.731 seconds.
- Final checkpoint eval-loss pattern:
  - `hard_target_trace` had the strongest loss profile among trained adapters.
  - `uniform4_trace` was close.
  - `hard_target_no_trace` and `hard_target_shuffled_trace` were much weaker, indicating that trace quality mattered during training.

Next step: run the final 39-job evaluation matrix.

### Final Evaluation

- Ran `scripts/run_final_evaluations.py --suite all`.
- Completed 39/39 evaluation jobs.
- Wrote per-job JSON results under `reports/final/`.
- Wrote final job manifest at `reports/final/final_evaluation_jobs.json`.
- Overall recombination-holdout results:
  - `frozen_trace`: 0/60.
  - `uniform2_trace`: 17/60.
  - `uniform4_trace`: 15/60.
  - `hard_target_trace`: 20/60.
  - `hard_target_seen_preserving_trace`: 15/60.
  - `easy_target_control_trace`: 19/60.
  - `modulo16_trace`: 20/60.
  - `length16_trace`: 12/60.
  - `tuple16_trace`: 17/60.
  - `hard_target_no_trace`: 5/60.
  - `hard_target_shuffled_trace`: 6/60.
  - `hard_target_trace` with no-trace prompt: 5/60.
  - `hard_target_trace` with shuffled-trace prompt: 4/60.
- Main target-family holdout counts:
  - `hard_target_trace`: `length_contains_code` 3/12, `modulo_sum_label` 0/12, `tuple_branch_label` 2/12.
  - `easy_target_control_trace`: `length_contains_code` 1/12, `modulo_sum_label` 0/12, `tuple_branch_label` 2/12.
  - `modulo16_trace`: `length_contains_code` 0/12, `modulo_sum_label` 5/12, `tuple_branch_label` 0/12.
  - `length16_trace`: `length_contains_code` 3/12, `modulo_sum_label` 0/12, `tuple_branch_label` 0/12.
  - `tuple16_trace`: `length_contains_code` 0/12, `modulo_sum_label` 2/12, `tuple_branch_label` 2/12.
  - `hard_target_no_trace`: 0/12 for all three target families.
  - `hard_target_shuffled_trace`: 0/12 for all three target families.
- Responsive-control holdout behavior was strong and often dominated aggregate scores:
  - `easy_target_control_trace` reached 12/12 on `sorted_contains_count` and 4/12 on `sorted_tuple_affine`.
  - `hard_target_trace` reached 12/12 on `sorted_contains_count` and 3/12 on `sorted_tuple_affine`.
  - `modulo16_trace` reached 8/12 on `sorted_contains_count` and 7/12 on `sorted_tuple_affine`.

### Report Generation

- Ran `scripts/make_report.py`.
- Wrote `reports/targeted_bridge_allocation_report.md`.
- Wrote CSV summaries:
  - `reports/final_results.csv`
  - `reports/final_results_by_family.csv`
  - `reports/final_results_by_factor.csv`
- Wrote figures:
  - `figures/final_repair_by_condition_split.png`
  - `figures/recombination_holdout_by_family.png`
- Validated final manifest:
  - 39 jobs.
  - all statuses `completed`.
  - all result JSON paths exist.
  - `reports/final_results.csv` has 39 rows.

### Artifact Split Verification

- Compact experiment directory: 24 MB.
- Large artifact directory: 13 GB.
- Confirmed no `.safetensors`, `.bin`, `.pt`, or `.pth` files in `experiments/targeted_bridge_allocation/`.
- Large artifact tree contains:
  - 10 adapter directories.
  - 410 files.
  - 30 checkpoint directories.
- Wrote `reports/large_artifacts_manifest.md`.

### Interpretation

- `hard_target_trace` is the strongest allocation choice for balanced recombination holdout behavior: it tied the top aggregate result at 20/60 and produced target-family successes on length and tuple while preserving strong responsive-control behavior.
- `easy_target_control_trace` nearly matched the aggregate at 19/60, but most of its holdout score came from responsive-control families. It did not improve modulo and only added one length success.
- `modulo16_trace` matched the top aggregate score and moved `modulo_sum_label` to 5/12, but it did not transfer to length or tuple.
- `length16_trace` moved `length_contains_code` to 3/12 but lowered aggregate holdout to 12/60.
- `tuple16_trace` did not improve `tuple_branch_label` beyond 2/12.
- Trace alignment is essential. Removing traces, shuffling traces at inference, training without traces, and training on shuffled traces all collapsed target-family holdout repair.
- The next high-impact direction is not simply increasing single-family bridge intensity. A better follow-up would preserve aligned traces while changing the training mix to support target-family mechanisms without sacrificing responsive-control coverage.
