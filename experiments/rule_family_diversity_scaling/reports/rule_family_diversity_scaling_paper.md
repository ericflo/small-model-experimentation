# Rule-Family Diversity Scaling

Date: 2026-06-21

## Abstract

This experiment tests whether increasing rule-family diversity in trace-conditioned repair training improves transfer to unseen rule structures when the total number of training records is fixed. Three trace-conditioned LoRA adapters were trained on 240 examples each, using 3, 6, or 12 rule families. The evaluation separates base-IID repair, format holdout, and fully held-out rule-family transfer. Controls test frozen-model behavior, no-trace training, shuffled-trace training, and prompt-time trace ablations.

## Design

- Base model: `Qwen/Qwen2.5-Coder-3B-Instruct`.
- Revision: `488639f1ff808d1d3d0ba301aef8c11461451ec5`.
- Training method: QLoRA, rank 32, alpha 64, dropout 0.05.
- Training budget: 240 records per diversity scale, 3 epochs.
- Evaluation metric: `repair@1`, requiring both visible and hidden tests to pass.
- Validation splits: 36 base-IID records, 36 format-holdout records, and 48 held-out rule-family records.

Training family counts:

- 3-family scale: 80 records per family.
- 6-family scale: 40 records per family.
- 12-family scale: 20 records per family.

Held-out rule families:

- `parity_offset_holdout`
- `quadratic_shift_holdout`
- `tuple_max_holdout`
- `sorted_join_holdout`

## Core Results

| Condition | Base IID | Format Holdout | Rule Holdout |
| --- | --- | --- | --- |
| Frozen trace | 0.0% (0/36) | 0.0% (0/36) | 0.0% (0/48) |
| 3 families, trace | 94.4% (34/36) | 50.0% (18/36) | 0.0% (0/48) |
| 6 families, trace | 77.8% (28/36) | 52.8% (19/36) | 4.2% (2/48) |
| 12 families, trace | 86.1% (31/36) | 44.4% (16/36) | 29.2% (14/48) |
| 12 families, no trace train/eval | 0.0% (0/36) | 0.0% (0/36) | 2.1% (1/48) |
| 12 families, shuffled trace train | 0.0% (0/36) | 0.0% (0/36) | 2.1% (1/48) |

## Diversity Scale Results

| Condition | Base IID | Format Holdout | Rule Holdout |
| --- | --- | --- | --- |
| 3 families, trace | 94.4% (34/36) | 50.0% (18/36) | 0.0% (0/48) |
| 6 families, trace | 77.8% (28/36) | 52.8% (19/36) | 4.2% (2/48) |
| 12 families, trace | 86.1% (31/36) | 44.4% (16/36) | 29.2% (14/48) |

## Trace Control And Ablation Results

| Condition | Base IID | Format Holdout | Rule Holdout |
| --- | --- | --- | --- |
| 12 families, trace | 86.1% (31/36) | 44.4% (16/36) | 29.2% (14/48) |
| 12-family trace adapter, no trace prompt | 0.0% (0/36) | 0.0% (0/36) | 0.0% (0/48) |
| 12-family trace adapter, shuffled trace prompt | 0.0% (0/36) | 0.0% (0/36) | 4.2% (2/48) |

## Held-Out Rule Results By Family

| Condition | Family | repair@1 | Successes |
| --- | --- | --- | --- |
| 3 families, trace | `parity_offset_holdout` | 0.0% | 0/12 |
| 3 families, trace | `quadratic_shift_holdout` | 0.0% | 0/12 |
| 3 families, trace | `sorted_join_holdout` | 0.0% | 0/12 |
| 3 families, trace | `tuple_max_holdout` | 0.0% | 0/12 |
| 6 families, trace | `parity_offset_holdout` | 0.0% | 0/12 |
| 6 families, trace | `quadratic_shift_holdout` | 16.7% | 2/12 |
| 6 families, trace | `sorted_join_holdout` | 0.0% | 0/12 |
| 6 families, trace | `tuple_max_holdout` | 0.0% | 0/12 |
| 12 families, trace | `parity_offset_holdout` | 0.0% | 0/12 |
| 12 families, trace | `quadratic_shift_holdout` | 25.0% | 3/12 |
| 12 families, trace | `sorted_join_holdout` | 91.7% | 11/12 |
| 12 families, trace | `tuple_max_holdout` | 0.0% | 0/12 |

## Interpretation

- Best held-out rule-family repair among the diversity-scale adapters was `scale12_trace` at 29.2% (14/48).
- Best format-holdout repair among the diversity-scale adapters was `scale6_trace` at 52.8% (19/36).
- Best base-IID repair among the diversity-scale adapters was `scale3_trace` at 94.4% (34/36).
- On held-out rule families, the 12-family trace adapter scored 29.2%, while the no-trace control scored 2.1% and the shuffled-trace-trained control scored 2.1%.

The central comparison is the diversity-scale curve on the rule-holdout split. A useful positive result is not simply high base-IID repair; it is improved rule-holdout repair under the same 240-record training budget. The control rows indicate whether any held-out repair depends on valid trace evidence or can be explained by patch priors learned from the training distribution.

## Figures

- `experiments/rule_family_diversity_scaling/figures/final_repair_by_condition_split.png`
- `experiments/rule_family_diversity_scaling/figures/diversity_scale_curve.png`
- `experiments/rule_family_diversity_scaling/figures/scale12_trace_ablation.png`

## Artifacts

- Dataset manifest: `experiments/rule_family_diversity_scaling/data/dataset_manifest.json`.
- Final JSON results: `experiments/rule_family_diversity_scaling/reports/final/`.
- CSV summaries: `experiments/rule_family_diversity_scaling/reports/*.csv`.
- Figures: `experiments/rule_family_diversity_scaling/figures/`.
- Large adapter artifacts: `large_artifacts/rule_family_diversity_scaling/models/`.
