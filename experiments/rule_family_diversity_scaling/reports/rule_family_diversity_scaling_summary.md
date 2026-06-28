# Rule-Family Diversity Scaling Summary

- Best held-out rule-family repair among the diversity-scale adapters was `scale12_trace` at 29.2% (14/48).
- Best format-holdout repair among the diversity-scale adapters was `scale6_trace` at 52.8% (19/36).
- Best base-IID repair among the diversity-scale adapters was `scale3_trace` at 94.4% (34/36).
- On held-out rule families, the 12-family trace adapter scored 29.2%, while the no-trace control scored 2.1% and the shuffled-trace-trained control scored 2.1%.

Core repair@1:

| Condition | Base IID | Format Holdout | Rule Holdout |
| --- | --- | --- | --- |
| Frozen trace | 0.0% (0/36) | 0.0% (0/36) | 0.0% (0/48) |
| 3 families, trace | 94.4% (34/36) | 50.0% (18/36) | 0.0% (0/48) |
| 6 families, trace | 77.8% (28/36) | 52.8% (19/36) | 4.2% (2/48) |
| 12 families, trace | 86.1% (31/36) | 44.4% (16/36) | 29.2% (14/48) |
| 12 families, no trace train/eval | 0.0% (0/36) | 0.0% (0/36) | 2.1% (1/48) |
| 12 families, shuffled trace train | 0.0% (0/36) | 0.0% (0/36) | 2.1% (1/48) |

The compact experiment package is `experiments/rule_family_diversity_scaling/`. Adapter weights and checkpoints are split out under `large_artifacts/rule_family_diversity_scaling/models/`.
