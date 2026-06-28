# Feature-Factorized Rule Diversity

## Question

Does held-out rule repair improve more when training examples cover isolated primitive factors, analogous multi-factor compositions, or a fixed-budget mixture of both?

## Design

- Three trace-trained adapters use the same 240-record budget: singleton factors only, composite factors only, and a mixed singleton/composite allocation.
- Two controls use the mixed allocation with traces removed or trace outputs shuffled during training.
- Final evaluation uses singleton IID, composite IID, and recombination holdout splits.
- Checkpoints and adapter weights are stored outside this experiment directory under `large_artifacts/feature_factorized_rule_diversity/models/`.

## Overall Results

| Condition | Singleton IID | Composite IID | Recombination Holdout |
| --- | --- | --- | --- |
| Frozen trace | 0.0% (0/40) | 0.0% (0/36) | 0.0% (0/60) |
| Singleton factors, trace | 85.0% (34/40) | 22.2% (8/36) | 20.0% (12/60) |
| Composite factors, trace | 10.0% (4/40) | 91.7% (33/36) | 23.3% (14/60) |
| Mixed factors, trace | 57.5% (23/40) | 69.4% (25/36) | 21.7% (13/60) |
| Mixed factors, no trace train/eval | 5.0% (2/40) | 16.7% (6/36) | 3.3% (2/60) |
| Mixed factors, shuffled trace train | 5.0% (2/40) | 13.9% (5/36) | 6.7% (4/60) |

## Prompt Ablations

| Condition | Singleton IID | Composite IID | Recombination Holdout |
| --- | --- | --- | --- |
| Mixed factors, trace | 57.5% (23/40) | 69.4% (25/36) | 21.7% (13/60) |
| Mixed trace adapter, no trace prompt | 5.0% (2/40) | 13.9% (5/36) | 0.0% (0/60) |
| Mixed trace adapter, shuffled trace prompt | 5.0% (2/40) | 8.3% (3/36) | 1.7% (1/60) |

## Recombination Holdout By Family

| Family | Singleton factors, trace | Composite factors, trace | Mixed factors, trace |
| --- | --- | --- | --- |
| contains_length_code_holdout | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| parity_offset_holdout | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| sorted_join_holdout | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) |
| sum_parity_shift_holdout | 0.0% (0/12) | 16.7% (2/12) | 8.3% (1/12) |
| tuple_max_label_holdout | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |

## Mixed Trace Holdout By Factor

| Factor | repair@1 |
| --- | --- |
| aggregation | 4.2% (1/24) |
| arithmetic | 2.1% (1/48) |
| branching | 2.1% (1/48) |
| length | 0.0% (0/12) |
| modulo | 4.2% (1/24) |
| sequence_iteration | 100.0% (12/12) |
| sorting | 100.0% (12/12) |
| string_format | 33.3% (12/36) |
| string_match | 0.0% (0/12) |
| string_normalization | 100.0% (12/12) |
| tuple_access | 0.0% (0/12) |

## Readout

- Best core recombination result: Composite factors, trace at 23.3% (14/60).
- Mixed trace vs controls on recombination: trace 21.7%, no trace 3.3%, shuffled-trace train 6.7%.

## Figures

- `experiments/feature_factorized_rule_diversity/figures/final_repair_by_condition_split.png`
- `experiments/feature_factorized_rule_diversity/figures/recombination_holdout_by_family.png`

## Artifact Layout

- Compact artifacts: `experiments/feature_factorized_rule_diversity/`.
- Large artifacts: `large_artifacts/feature_factorized_rule_diversity/`.
- Dataset manifest: `experiments/feature_factorized_rule_diversity/data/dataset_manifest.json`.
- Evaluation manifest: `experiments/feature_factorized_rule_diversity/reports/final/final_evaluation_jobs.json`.
