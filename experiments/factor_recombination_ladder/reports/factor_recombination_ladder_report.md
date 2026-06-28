# Factor Recombination Ladder

## Question

Can trace-conditioned repair learn reusable factor recombination when specific factor-pair cells are held out from training?

## Design

- The training set contains 12 seen rule families with 240 total records per condition.
- The recombination split contains five held-out factor-pair cells absent from training.
- Core conditions compare frozen trace prompting, aligned trace training, no-trace training, shuffled-trace training, and factor-labelled trace training.
- Prompt ablations test whether trained trace adapters depend on trace content and factor labels at inference time.
- Checkpoints and adapter weights are stored outside this experiment directory under `large_artifacts/factor_recombination_ladder/models/`.

## Overall Results

| Condition | Seen-Combination IID | Format Shift | Recombination Holdout |
| --- | --- | --- | --- |
| Frozen trace | 0.0% (0/36) | 0.0% (0/36) | 0.0% (0/60) |
| Trace ladder | 80.6% (29/36) | 63.9% (23/36) | 8.3% (5/60) |
| No-trace ladder | 33.3% (12/36) | 25.0% (9/36) | 10.0% (6/60) |
| Shuffled-trace ladder | 27.8% (10/36) | 30.6% (11/36) | 8.3% (5/60) |
| Factor-labelled trace ladder | 80.6% (29/36) | 72.2% (26/36) | 8.3% (5/60) |

## Prompt Ablations

| Condition | Seen-Combination IID | Format Shift | Recombination Holdout |
| --- | --- | --- | --- |
| Trace ladder | 80.6% (29/36) | 63.9% (23/36) | 8.3% (5/60) |
| Trace ladder, no trace prompt | 16.7% (6/36) | 19.4% (7/36) | 8.3% (5/60) |
| Trace ladder, shuffled trace prompt | 0.0% (0/36) | 2.8% (1/36) | 6.7% (4/60) |
| Factor-labelled trace ladder | 80.6% (29/36) | 72.2% (26/36) | 8.3% (5/60) |
| Labelled adapter, labels removed | 75.0% (27/36) | 69.4% (25/36) | 5.0% (3/60) |
| Labelled adapter, no trace prompt | 22.2% (8/36) | 16.7% (6/36) | 8.3% (5/60) |
| Labelled adapter, shuffled trace prompt | 13.9% (5/36) | 8.3% (3/36) | 6.7% (4/60) |

## Recombination Holdout By Family

| Family | Trace ladder | Factor-labelled trace ladder | No-trace ladder | Shuffled-trace ladder |
| --- | --- | --- | --- | --- |
| length_contains_code | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| modulo_sum_label | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sorted_contains_count | 8.3% (1/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sorted_tuple_affine | 33.3% (4/12) | 33.3% (4/12) | 50.0% (6/12) | 41.7% (5/12) |
| tuple_branch_label | 0.0% (0/12) | 8.3% (1/12) | 0.0% (0/12) | 0.0% (0/12) |

## Trace Ladder Holdout By Factor

| Factor | repair@1 |
| --- | --- |
| aggregation | 4.2% (1/24) |
| arithmetic | 8.3% (4/48) |
| branching | 0.0% (0/36) |
| length | 0.0% (0/12) |
| modulo | 0.0% (0/12) |
| ordering | 20.8% (5/24) |
| sequence_iteration | 8.3% (1/12) |
| string_format | 2.1% (1/48) |
| string_match | 4.2% (1/24) |
| string_normalization | 8.3% (1/12) |
| tuple_access | 16.7% (4/24) |

## Readout

- Best core recombination result: No-trace ladder at 10.0% (6/60).
- Factor labels delta on recombination: trace 8.3% (5/60) vs labelled trace 8.3% (5/60).

## Figures

- `experiments/factor_recombination_ladder/figures/final_repair_by_condition_split.png`
- `experiments/factor_recombination_ladder/figures/recombination_holdout_by_family.png`

## Artifact Layout

- Compact artifacts: `experiments/factor_recombination_ladder/`.
- Large artifacts: `large_artifacts/factor_recombination_ladder/`.
- Dataset manifest: `experiments/factor_recombination_ladder/data/dataset_manifest.json`.
- Evaluation manifest: `experiments/factor_recombination_ladder/reports/final/final_evaluation_jobs.json`.
