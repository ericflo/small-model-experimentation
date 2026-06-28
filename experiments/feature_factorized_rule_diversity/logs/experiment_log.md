# Experiment Log

## 2026-06-21

- Created standalone experiment directory at `experiments/feature_factorized_rule_diversity/`.
- Created large-artifact directory at `large_artifacts/feature_factorized_rule_diversity/models/`.
- Defined the central question: compare fixed-budget primitive factor coverage, analogous composition coverage, and a mixed allocation for trace-conditioned held-out recombination repair.
- Implemented `scripts/build_factorized_dataset.py`.
  - The builder writes three 240-record training sets.
  - It writes three evaluation splits: singleton IID, composite IID, and recombination holdout.
  - Each generated record is validated before writing:
    - the wrong implementation fails visible tests,
    - the target corrective diff applies,
    - the target implementation passes visible and hidden tests,
    - hidden inputs are disjoint from visible inputs,
    - visible expected outputs appear in the failing trace.
- Implemented `scripts/eval_factorized.py` with summary outputs by split, family, and factor tag.
- Implemented `scripts/run_training.py` so all adapter training jobs and hyperparameters are captured in `reports/training/training_jobs.json`.
- Implemented `scripts/run_final_evaluations.py` for the full final evaluation matrix.
- Implemented `scripts/make_report.py` for CSV summaries, plots, markdown report, and large-artifact manifest refresh.

Next step: build and validate the dataset, then train the five planned adapters.

### Dataset Build Pass 1

- Started dataset generation with full pytest validation.
- Interrupted after the run remained silent for too long during the first split.
- Diagnosis: validation was active, but the builder had no progress logging. The test template also used a looped pytest test, which can hide later visible counterexamples after the first failure.
- Fix applied:
  - converted generated tests to `pytest.mark.parametrize` cases,
  - added per-family progress logging to stderr.

### Dataset Build Pass 2

- Completed dataset generation and validation.
- Wrote 856 total records:
  - `train_singletons`: 240 records, 10 singleton families x 24 records.
  - `train_composites`: 240 records, 6 composite families x 40 records.
  - `train_mixed`: 240 records, 8 mixed families x 30 records.
  - `val_singleton_iid`: 40 records.
  - `val_composite_iid`: 36 records.
  - `val_recombination_holdout`: 60 records, 5 held-out recombination families x 12 records.
- Confirmed compact experiment directory size after dataset build: about 13 MB.
- Confirmed `large_artifacts/feature_factorized_rule_diversity/` remains empty before training.

Next step: train five LoRA adapters: singleton trace, composite trace, mixed trace, mixed no-trace, and mixed shuffled-trace.

### Training

- Completed all five planned LoRA training jobs.
- Training jobs and exact commands are recorded in `reports/training/training_jobs.json`.
- Console output is recorded in `run_logs/training_console.log`.
- Adapter directories:
  - `large_artifacts/feature_factorized_rule_diversity/models/singletons_trace_lora`
  - `large_artifacts/feature_factorized_rule_diversity/models/composites_trace_lora`
  - `large_artifacts/feature_factorized_rule_diversity/models/mixed_trace_lora`
  - `large_artifacts/feature_factorized_rule_diversity/models/mixed_no_trace_lora`
  - `large_artifacts/feature_factorized_rule_diversity/models/mixed_shuffled_trace_lora`
- Approximate adapter directory size after training: 1.3 GB each.
- Final checkpoint eval losses:
  - singleton trace on singleton IID: about 0.0071.
  - composite trace on composite IID: about 0.0018.
  - mixed trace on recombination holdout: about 0.0886.
  - mixed no-trace on recombination holdout: about 0.2108.
  - mixed shuffled-trace on recombination holdout: about 0.1782.

Next step: run final generation evaluations for frozen baseline, trace adapters, controls, and prompt ablations.

### Final Generation Evaluation

- Completed the full 24-job final evaluation suite.
- Final evaluation jobs and result paths are recorded in `reports/final/final_evaluation_jobs.json`.
- Console output is recorded in `run_logs/final_evaluation_console.log`.
- Generated CSV summaries:
  - `reports/final_results.csv`
  - `reports/final_results_by_family.csv`
  - `reports/final_results_by_factor.csv`
- Generated report and figures:
  - `reports/feature_factorized_rule_diversity_report.md`
  - `figures/final_repair_by_condition_split.png`
  - `figures/recombination_holdout_by_family.png`

Core repair@1 results:

| Condition | Singleton IID | Composite IID | Recombination Holdout |
| --- | --- | --- | --- |
| Frozen trace | 0/40 | 0/36 | 0/60 |
| Singleton factors, trace | 34/40 | 8/36 | 12/60 |
| Composite factors, trace | 4/40 | 33/36 | 14/60 |
| Mixed factors, trace | 23/40 | 25/36 | 13/60 |
| Mixed factors, no trace train/eval | 2/40 | 6/36 | 2/60 |
| Mixed factors, shuffled trace train | 2/40 | 5/36 | 4/60 |

Prompt ablations for the mixed trace adapter:

| Prompt condition | Singleton IID | Composite IID | Recombination Holdout |
| --- | --- | --- | --- |
| Normal aligned trace prompt | 23/40 | 25/36 | 13/60 |
| No-trace prompt | 2/40 | 5/36 | 0/60 |
| Shuffled-trace prompt | 2/40 | 3/36 | 1/60 |

Recombination holdout family pattern for the three main trace adapters:

| Family | Singleton trace | Composite trace | Mixed trace |
| --- | --- | --- | --- |
| contains_length_code_holdout | 0/12 | 0/12 | 0/12 |
| parity_offset_holdout | 0/12 | 0/12 | 0/12 |
| sorted_join_holdout | 12/12 | 12/12 | 12/12 |
| sum_parity_shift_holdout | 0/12 | 2/12 | 1/12 |
| tuple_max_label_holdout | 0/12 | 0/12 | 0/12 |

Interpretation:

- The strongest in-distribution behavior is narrow specialization: singleton trace is best on singleton IID, and composite trace is best on composite IID.
- The fixed-budget mixed trace adapter improves breadth relative to single-domain adapters, but it does not improve the recombination holdout. It reaches 13/60, slightly below composite trace at 14/60 and only slightly above singleton trace at 12/60.
- The apparent recombination success is mostly not broad recombination. All three trace adapters solve `sorted_join_holdout` perfectly, but they fail three of five holdout families completely. Composite trace gets only 2/12 on `sum_parity_shift_holdout`; mixed trace gets only 1/12.
- Aligned traces are clearly important. Removing traces during training collapses mixed performance to 2/40, 6/36, and 2/60. Shuffling training traces also collapses to 2/40, 5/36, and 4/60.
- The aligned mixed trace adapter is also highly dependent on the inference prompt contract. With no trace prompt it drops to 2/40, 5/36, and 0/60. With shuffled trace prompts it drops to 2/40, 3/36, and 1/60.

Decision readout:

- This experiment does not support simple fixed-budget mixing as the next scaling direction for broad recombination.
- The most useful positive signal is that aligned traces carry real supervision. The most important failure is that traces and examples do not yet teach reusable factor abstractions for length/string-match/tuple/modulo recombinations.
- The next experiment should preserve aligned trace supervision, but change the data design toward factor-balanced recombination coverage and trace consistency diagnostics instead of simply mixing singleton and composite examples.

### Artifact Verification

- Compact experiment directory: `experiments/feature_factorized_rule_diversity/`, about 19 MB after reports and figures.
- Large artifact directory: `large_artifacts/feature_factorized_rule_diversity/`, about 6.4 GB total.
- Adapter/checkpoint directories are intentionally outside the compact experiment package:
  - `large_artifacts/feature_factorized_rule_diversity/models/singletons_trace_lora`
  - `large_artifacts/feature_factorized_rule_diversity/models/composites_trace_lora`
  - `large_artifacts/feature_factorized_rule_diversity/models/mixed_trace_lora`
  - `large_artifacts/feature_factorized_rule_diversity/models/mixed_no_trace_lora`
  - `large_artifacts/feature_factorized_rule_diversity/models/mixed_shuffled_trace_lora`
- Large artifact manifest refreshed at `large_artifacts_manifest.md`.
