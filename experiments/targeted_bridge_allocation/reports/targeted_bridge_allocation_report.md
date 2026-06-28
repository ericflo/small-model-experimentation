# Targeted Bridge Allocation

## Question

Does fixed-budget targeted bridge allocation improve trace-conditioned recombination repair compared with uniform bridge allocation?

## Design

- Every trained condition uses a fixed 240-record budget.
- Uniform baselines allocate exact bridge examples evenly across five held-out recombination families.
- Targeted conditions concentrate bridge examples on three target families while retaining some bridges for two responsive-control families.
- A seen-preserving targeted condition keeps total bridge count equal to the uniform4 baseline.
- An easy-target control spends the same bridge budget as the main targeted condition, but concentrates it on the responsive-control families.
- Three light single-family probes raise one target family to 16 bridge examples while keeping every other held-out family at 2.
- No-trace and shuffled-trace controls on the main targeted allocation test whether any allocation effect depends on aligned trace evidence.
- Checkpoints and adapter weights are stored outside this experiment directory under `large_artifacts/targeted_bridge_allocation/models/`.

## Allocation Plans

| Plan | Bridge total | Seen total | modulo_sum_label | length_contains_code | tuple_branch_label | sorted_contains_count | sorted_tuple_affine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Uniform 2 trace | 10 | 230 | 2 | 2 | 2 | 2 | 2 |
| Uniform 4 trace | 20 | 220 | 4 | 4 | 4 | 4 | 4 |
| Hard-target trace | 28 | 212 | 8 | 8 | 8 | 2 | 2 |
| Hard-target seen-preserving trace | 20 | 220 | 6 | 6 | 6 | 1 | 1 |
| Easy-target control trace | 28 | 212 | 2 | 2 | 2 | 11 | 11 |
| Modulo-16 trace | 24 | 216 | 16 | 2 | 2 | 2 | 2 |
| Length-16 trace | 24 | 216 | 2 | 16 | 2 | 2 | 2 |
| Tuple-16 trace | 24 | 216 | 2 | 2 | 16 | 2 | 2 |

## Overall Results

| Condition | Seen-Combination IID | Format Shift | Recombination Holdout |
| --- | --- | --- | --- |
| Frozen trace | 0.0% (0/36) | 0.0% (0/36) | 0.0% (0/60) |
| Uniform 2 trace | 75.0% (27/36) | 72.2% (26/36) | 28.3% (17/60) |
| Uniform 4 trace | 72.2% (26/36) | 83.3% (30/36) | 25.0% (15/60) |
| Hard-target trace | 77.8% (28/36) | 69.4% (25/36) | 33.3% (20/60) |
| Hard-target seen-preserving trace | 75.0% (27/36) | 72.2% (26/36) | 25.0% (15/60) |
| Easy-target control trace | 80.6% (29/36) | 80.6% (29/36) | 31.7% (19/60) |
| Modulo-16 trace | 83.3% (30/36) | 75.0% (27/36) | 33.3% (20/60) |
| Length-16 trace | 80.6% (29/36) | 77.8% (28/36) | 20.0% (12/60) |
| Tuple-16 trace | 75.0% (27/36) | 69.4% (25/36) | 28.3% (17/60) |
| Hard-target no-trace | 27.8% (10/36) | 19.4% (7/36) | 8.3% (5/60) |
| Hard-target shuffled-trace train | 30.6% (11/36) | 25.0% (9/36) | 10.0% (6/60) |

## Allocation Comparison

| Condition | Seen-Combination IID | Format Shift | Recombination Holdout |
| --- | --- | --- | --- |
| Uniform 2 trace | 75.0% (27/36) | 72.2% (26/36) | 28.3% (17/60) |
| Uniform 4 trace | 72.2% (26/36) | 83.3% (30/36) | 25.0% (15/60) |
| Hard-target trace | 77.8% (28/36) | 69.4% (25/36) | 33.3% (20/60) |
| Hard-target seen-preserving trace | 75.0% (27/36) | 72.2% (26/36) | 25.0% (15/60) |
| Easy-target control trace | 80.6% (29/36) | 80.6% (29/36) | 31.7% (19/60) |

## Light Single-Family Probes

| Condition | Seen-Combination IID | Format Shift | Recombination Holdout |
| --- | --- | --- | --- |
| Uniform 2 trace | 75.0% (27/36) | 72.2% (26/36) | 28.3% (17/60) |
| Modulo-16 trace | 83.3% (30/36) | 75.0% (27/36) | 33.3% (20/60) |
| Length-16 trace | 80.6% (29/36) | 77.8% (28/36) | 20.0% (12/60) |
| Tuple-16 trace | 75.0% (27/36) | 69.4% (25/36) | 28.3% (17/60) |

## Prompt Ablations

| Condition | Seen-Combination IID | Format Shift | Recombination Holdout |
| --- | --- | --- | --- |
| Hard-target trace | 77.8% (28/36) | 69.4% (25/36) | 33.3% (20/60) |
| Hard-target trace, no-trace prompt | 19.4% (7/36) | 16.7% (6/36) | 8.3% (5/60) |
| Hard-target trace, shuffled-trace prompt | 8.3% (3/36) | 5.6% (2/36) | 6.7% (4/60) |

## Recombination Holdout By Family

| Family | Uniform 2 trace | Uniform 4 trace | Hard-target trace | Hard-target seen-preserving trace | Easy-target control trace | Modulo-16 trace | Length-16 trace | Tuple-16 trace |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| length_contains_code | 0.0% (0/12) | 16.7% (2/12) | 25.0% (3/12) | 0.0% (0/12) | 8.3% (1/12) | 0.0% (0/12) | 25.0% (3/12) | 0.0% (0/12) |
| modulo_sum_label | 0.0% (0/12) | 8.3% (1/12) | 0.0% (0/12) | 41.7% (5/12) | 0.0% (0/12) | 41.7% (5/12) | 0.0% (0/12) | 16.7% (2/12) |
| sorted_contains_count | 100.0% (12/12) | 58.3% (7/12) | 100.0% (12/12) | 66.7% (8/12) | 100.0% (12/12) | 66.7% (8/12) | 41.7% (5/12) | 58.3% (7/12) |
| sorted_tuple_affine | 41.7% (5/12) | 41.7% (5/12) | 25.0% (3/12) | 16.7% (2/12) | 33.3% (4/12) | 58.3% (7/12) | 33.3% (4/12) | 50.0% (6/12) |
| tuple_branch_label | 0.0% (0/12) | 0.0% (0/12) | 16.7% (2/12) | 0.0% (0/12) | 16.7% (2/12) | 0.0% (0/12) | 0.0% (0/12) | 16.7% (2/12) |

## Hard-Target Holdout By Factor

| Factor | repair@1 |
| --- | --- |
| aggregation | 50.0% (12/24) |
| arithmetic | 16.7% (8/48) |
| branching | 13.9% (5/36) |
| length | 25.0% (3/12) |
| modulo | 0.0% (0/12) |
| ordering | 62.5% (15/24) |
| sequence_iteration | 100.0% (12/12) |
| string_format | 35.4% (17/48) |
| string_match | 62.5% (15/24) |
| string_normalization | 100.0% (12/12) |
| tuple_access | 20.8% (5/24) |

## Readout

- Best core recombination result: Hard-target trace at 33.3% (20/60).
- Hard-target vs uniform2: 33.3% (20/60) vs 28.3% (17/60), delta 5.0%.
- Seen-preserving target vs uniform4: 25.0% (15/60) vs 25.0% (15/60), delta 0.0%.
- Hard-target vs easy-target budget control: 33.3% (20/60) vs 31.7% (19/60), delta 1.7%.

## Figures

- `experiments/targeted_bridge_allocation/figures/final_repair_by_condition_split.png`
- `experiments/targeted_bridge_allocation/figures/recombination_holdout_by_family.png`

## Artifact Layout

- Compact artifacts: `experiments/targeted_bridge_allocation/`.
- Large artifacts: `large_artifacts/targeted_bridge_allocation/`.
- Dataset manifest: `experiments/targeted_bridge_allocation/data/dataset_manifest.json`.
- Evaluation manifest: `experiments/targeted_bridge_allocation/reports/final/final_evaluation_jobs.json`.
