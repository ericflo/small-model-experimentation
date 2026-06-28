# Bridge-Dose Recombination Curriculum

## Question

How many exact bridge examples are needed before trace-conditioned repair generalizes across withheld factor-pair cells?

## Design

- Every trained condition uses a fixed 240-record budget.
- Dose conditions add `k` examples from each withheld factor-pair family, with `k` in `{0, 1, 2, 4, 8}`.
- Seen-combination records are removed as bridge examples are added, so gains cannot come from a larger dataset.
- A near-miss focus control uses no exact withheld pairs but reallocates examples toward families sharing one primitive factor with the withheld pairs.
- Endpoint no-trace and shuffled-trace controls test whether any bridge effect depends on aligned trace evidence.
- Checkpoints and adapter weights are stored outside this experiment directory under `large_artifacts/bridge_dose_recombination_curriculum/models/`.

## Overall Results

| Condition | Seen-Combination IID | Format Shift | Recombination Holdout |
| --- | --- | --- | --- |
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

## Dose Curve

| Condition | Seen-Combination IID | Format Shift | Recombination Holdout |
| --- | --- | --- | --- |
| Dose 0 trace | 77.8% (28/36) | 80.6% (29/36) | 6.7% (4/60) |
| Dose 1 trace | 86.1% (31/36) | 80.6% (29/36) | 15.0% (9/60) |
| Dose 2 trace | 83.3% (30/36) | 75.0% (27/36) | 28.3% (17/60) |
| Dose 4 trace | 72.2% (26/36) | 77.8% (28/36) | 31.7% (19/60) |
| Dose 8 trace | 58.3% (21/36) | 80.6% (29/36) | 30.0% (18/60) |
| Near-miss focus trace | 83.3% (30/36) | 80.6% (29/36) | 8.3% (5/60) |

## Prompt Ablations

| Condition | Seen-Combination IID | Format Shift | Recombination Holdout |
| --- | --- | --- | --- |
| Dose 8 trace | 58.3% (21/36) | 80.6% (29/36) | 30.0% (18/60) |
| Dose 8 trace, no-trace prompt | 19.4% (7/36) | 13.9% (5/36) | 8.3% (5/60) |
| Dose 8 trace, shuffled-trace prompt | 5.6% (2/36) | 11.1% (4/36) | 6.7% (4/60) |

## Recombination Holdout By Family

| Family | Dose 0 trace | Dose 2 trace | Dose 8 trace | Near-miss focus trace | Dose 8 no-trace | Dose 8 shuffled-trace train |
| --- | --- | --- | --- | --- | --- | --- |
| length_contains_code | 0.0% (0/12) | 0.0% (0/12) | 33.3% (4/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| modulo_sum_label | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sorted_contains_count | 0.0% (0/12) | 100.0% (12/12) | 91.7% (11/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sorted_tuple_affine | 33.3% (4/12) | 33.3% (4/12) | 16.7% (2/12) | 41.7% (5/12) | 41.7% (5/12) | 41.7% (5/12) |
| tuple_branch_label | 0.0% (0/12) | 8.3% (1/12) | 8.3% (1/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |

## Dose 8 Holdout By Factor

| Factor | repair@1 |
| --- | --- |
| aggregation | 45.8% (11/24) |
| arithmetic | 14.6% (7/48) |
| branching | 13.9% (5/36) |
| length | 33.3% (4/12) |
| modulo | 0.0% (0/12) |
| ordering | 54.2% (13/24) |
| sequence_iteration | 91.7% (11/12) |
| string_format | 33.3% (16/48) |
| string_match | 62.5% (15/24) |
| string_normalization | 91.7% (11/12) |
| tuple_access | 12.5% (3/24) |

## Readout

- Best core recombination result: Dose 4 trace at 31.7% (19/60).
- Dose response from `k=0` to `k=8`: 6.7% (4/60) to 30.0% (18/60), delta 23.3%.
- Pair anchoring check: dose 8 trace 30.0% (18/60) vs near-miss focus 8.3% (5/60).

## Figures

- `experiments/bridge_dose_recombination_curriculum/figures/final_repair_by_condition_split.png`
- `experiments/bridge_dose_recombination_curriculum/figures/recombination_holdout_by_family.png`

## Artifact Layout

- Compact artifacts: `experiments/bridge_dose_recombination_curriculum/`.
- Large artifacts: `large_artifacts/bridge_dose_recombination_curriculum/`.
- Dataset manifest: `experiments/bridge_dose_recombination_curriculum/data/dataset_manifest.json`.
- Evaluation manifest: `experiments/bridge_dose_recombination_curriculum/reports/final/final_evaluation_jobs.json`.
