# Experiment Log

## 2026-06-21

- Created standalone experiment directory at `experiments/bridge_dose_recombination_curriculum/`.
- Created large-artifact directory at `large_artifacts/bridge_dose_recombination_curriculum/models/`.
- Defined the central question: how many exact bridge examples are needed before trace-conditioned repair generalizes across withheld factor-pair cells?
- Planned a fixed-budget bridge-dose curriculum:
  - every trained condition has 240 records,
  - bridge doses are `k = 0, 1, 2, 4, 8` examples per withheld factor-pair family,
  - adding bridge examples removes seen-combination examples so dataset size is constant,
  - a near-miss focus control has no exact withheld factor pairs but reallocates examples toward related primitive factors,
  - endpoint no-trace and shuffled-trace controls test whether any bridge effect depends on aligned trace evidence.
- Implemented the experiment package:
  - `scripts/build_bridge_dataset.py`
  - `scripts/eval_bridge.py`
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

- Ran `scripts/build_bridge_dataset.py` with seed `20260621`.
- Completed executable validation for all generated records.
- Wrote six 240-record training files:
  - `repair_train_dose0.jsonl`
  - `repair_train_dose1.jsonl`
  - `repair_train_dose2.jsonl`
  - `repair_train_dose4.jsonl`
  - `repair_train_dose8.jsonl`
  - `repair_train_near_miss_focus.jsonl`
- Wrote three fixed evaluation splits:
  - `repair_val_seen_iid.jsonl`: 36 records.
  - `repair_val_format_shift.jsonl`: 36 records.
  - `repair_val_recombination_holdout.jsonl`: 60 records.
- Verified bridge-count invariants from `data/dataset_manifest.json`:
  - `repair_train_dose0`: no exact bridge examples.
  - `repair_train_dose1`: 1 example for each withheld factor pair.
  - `repair_train_dose2`: 2 examples for each withheld factor pair.
  - `repair_train_dose4`: 4 examples for each withheld factor pair.
  - `repair_train_dose8`: 8 examples for each withheld factor pair.
  - `repair_train_near_miss_focus`: no exact bridge examples.
- Withheld factor pairs:
  - `aggregation+modulo`
  - `branching+tuple_access`
  - `length+string_match`
  - `ordering+string_match`
  - `ordering+tuple_access`
- Compact experiment directory after dataset build: about 11 MB.
- Large artifact directory remains empty before training.

Next step: train the dose curve adapters and endpoint controls.

### Training

- Ran `scripts/run_training.py --suite all`.
- Trained all 10 planned LoRA adapters with the shared configuration:
  - base model: `Qwen/Qwen2.5-Coder-3B-Instruct`
  - revision: `488639f1ff808d1d3d0ba301aef8c11461451ec5`
  - epochs: 3
  - learning rate: `1.5e-4`
  - LoRA rank/alpha/dropout: `32/64/0.05`
  - max sequence length: `3072`
  - gradient accumulation: `8`
  - checkpoint/eval cadence: every 30 optimizer steps
- Adapter outputs were written under `large_artifacts/bridge_dose_recombination_curriculum/models/`, not inside the compact experiment directory.
- All jobs completed successfully. Runtime per adapter was about 7 minutes.

Held-out recombination validation loss by epoch:

| adapter | epoch 1 | epoch 2 | epoch 3 |
| --- | ---: | ---: | ---: |
| `dose0_trace` | 0.1122 | 0.1312 | 0.1495 |
| `dose1_trace` | 0.08764 | 0.07808 | 0.07351 |
| `dose2_trace` | 0.07014 | 0.05168 | 0.05419 |
| `dose4_trace` | 0.05970 | 0.04071 | 0.04115 |
| `dose8_trace` | 0.05468 | 0.03958 | 0.03626 |
| `near_miss_focus_trace` | 0.1525 | 0.1286 | 0.1355 |
| `dose0_no_trace` | 0.1918 | 0.1806 | 0.1962 |
| `dose0_shuffled_trace` | 0.1790 | 0.1966 | 0.2040 |
| `dose8_no_trace` | 0.1201 | 0.1001 | 0.09461 |
| `dose8_shuffled_trace` | 0.1251 | 0.1056 | 0.09834 |

Interim observations before final exact-match evaluation:

- Zero-dose trace overfit the seen combinations by held-out loss: 0.1122 -> 0.1495.
- Exact bridge examples produced a monotonic endpoint improvement across the trace dose curve: dose 1, 2, 4, and 8 all beat dose 0, with dose 8 best at 0.03626.
- The near-miss focus control failed to substitute for exact bridge examples despite low training loss.
- Removing traces at dose 0 was much worse than coherent traces; shuffling trace lines was similarly poor.
- At dose 8, no-trace and shuffled-trace controls improved versus their dose 0 counterparts, but stayed far behind coherent trace. This suggests exact bridge coverage and coherent trace conditioning are complementary.

Next step: run the fixed final evaluation matrix across seen IID, format shift, and recombination holdout splits.

### Final Evaluation

- Ran `scripts/run_final_evaluations.py --suite all`.
- Completed all 39 planned final evaluation jobs.
- Wrote the evaluation manifest to `reports/final/final_evaluation_jobs.json`.
- Wrote aggregate CSV outputs:
  - `reports/final_results.csv`
  - `reports/final_results_by_family.csv`
  - `reports/final_results_by_factor.csv`
- Generated figures:
  - `figures/final_repair_by_condition_split.png`
  - `figures/recombination_holdout_by_family.png`

Overall repair@1 results:

| Condition | Seen IID | Format Shift | Recombination Holdout |
| --- | ---: | ---: | ---: |
| Frozen trace | 0.0% (0/36) | 0.0% (0/36) | 0.0% (0/60) |
| Dose 0 trace | 77.8% (28/36) | 80.6% (29/36) | 6.7% (4/60) |
| Dose 1 trace | 86.1% (31/36) | 80.6% (29/36) | 15.0% (9/60) |
| Dose 2 trace | 83.3% (30/36) | 75.0% (27/36) | 28.3% (17/60) |
| Dose 4 trace | 72.2% (26/36) | 77.8% (28/36) | 31.7% (19/60) |
| Dose 8 trace | 58.3% (21/36) | 80.6% (29/36) | 30.0% (18/60) |
| Near-miss focus trace | 83.3% (30/36) | 80.6% (29/36) | 8.3% (5/60) |
| Dose 0 no-trace | 27.8% (10/36) | 22.2% (8/36) | 5.0% (3/60) |
| Dose 0 shuffled-trace train | 25.0% (9/36) | 27.8% (10/36) | 6.7% (4/60) |
| Dose 8 no-trace | 30.6% (11/36) | 30.6% (11/36) | 8.3% (5/60) |
| Dose 8 shuffled-trace train | 19.4% (7/36) | 33.3% (12/36) | 8.3% (5/60) |

Prompt ablations on the dose 8 trace adapter:

| Prompt condition | Seen IID | Format Shift | Recombination Holdout |
| --- | ---: | ---: | ---: |
| Trace prompt | 58.3% (21/36) | 80.6% (29/36) | 30.0% (18/60) |
| No-trace prompt | 19.4% (7/36) | 13.9% (5/36) | 8.3% (5/60) |
| Shuffled-trace prompt | 5.6% (2/36) | 11.1% (4/36) | 6.7% (4/60) |

Recombination holdout by selected family:

| Family | Dose 0 trace | Dose 2 trace | Dose 8 trace | Near-miss focus trace | Dose 8 no-trace | Dose 8 shuffled-trace train |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| length_contains_code | 0/12 | 0/12 | 4/12 | 0/12 | 0/12 | 0/12 |
| modulo_sum_label | 0/12 | 0/12 | 0/12 | 0/12 | 0/12 | 0/12 |
| sorted_contains_count | 0/12 | 12/12 | 11/12 | 0/12 | 0/12 | 0/12 |
| sorted_tuple_affine | 4/12 | 4/12 | 2/12 | 5/12 | 5/12 | 5/12 |
| tuple_branch_label | 0/12 | 1/12 | 1/12 | 0/12 | 0/12 | 0/12 |

Interpretation:

- Exact bridge examples are the main lever. The trace dose curve moved recombination holdout from 4/60 at dose 0 to 9/60, 17/60, 19/60, and 18/60 at doses 1, 2, 4, and 8.
- Dose 2 is the practical frontier: it captures most of the recombination gain while retaining 30/36 seen-IID repairs. Dose 4 is best on holdout by two additional examples but costs seen-IID accuracy. Dose 8 adds no further holdout gain and sharply reduces seen-IID performance.
- Pair anchoring matters. The near-miss focus control retained strong seen-IID and format-shift performance but reached only 5/60 on recombination holdout.
- Coherent traces matter at both train time and prompt time. No-trace and shuffled-trace training stayed near floor on holdout, and removing or shuffling trace context at inference collapsed the dose 8 trace adapter.
- Gains are concentrated rather than uniform. `sorted_contains_count` became mostly solved under the bridge curriculum, while `modulo_sum_label` remained 0/12 across measured conditions and `tuple_branch_label` barely moved.

### Report Generation

- Ran `scripts/make_report.py`.
- Wrote the final markdown report to `reports/bridge_dose_recombination_curriculum_report.md`.
- Updated `README.md` with the final result snapshot, key output list, and download guidance.

### Artifact Split

- Compact experiment directory: `experiments/bridge_dose_recombination_curriculum/`.
- Large model artifact directory: `large_artifacts/bridge_dose_recombination_curriculum/models/`.
- All LoRA adapters and checkpoints were written to the large artifact directory.
- The compact directory is intended to be downloaded independently and should contain no `.safetensors`, `.bin`, `.pt`, or `.pth` files.

Final next-experiment implication:

- The most promising follow-up is not a larger global bridge dose. The useful next step is a targeted per-family bridge allocation experiment that starts from the dose 2 frontier, reallocates bridge budget toward persistent failure families (`modulo_sum_label`, `tuple_branch_label`, and `length_contains_code`), and preserves enough seen-combination coverage to avoid the dose 8 seen-IID regression.

### Final Verification

- `python -m py_compile experiments/bridge_dose_recombination_curriculum/scripts/*.py` completed successfully.
- Verified `reports/final/final_evaluation_jobs.json` contains 39 jobs and all have status `completed`.
- Verified `reports/final_results.csv` contains 39 rows and uses the expected `repair_at_1` metric column.
- Confirmed no `.safetensors`, `.bin`, `.pt`, or `.pth` files are present under `experiments/bridge_dose_recombination_curriculum/`.
- Confirmed no `__pycache__` directories remain after cleanup.
- Final directory sizes:
  - Compact experiment directory: 22 MB.
  - Large artifact directory: 13 GB.
- Large artifact file count under `large_artifacts/bridge_dose_recombination_curriculum/models/`: 80 files.

Status: complete. The standalone experiment directory is ready to download without large model artifacts; adapter weights and checkpoints remain split out under `large_artifacts/`.
