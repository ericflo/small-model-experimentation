# Qwen3.5-4B Deployable Information Ceiling Sweep

## Objective

This standalone diagnostic measures whether the hard low-information regime is limited by deployable information or by a trainable probe-selection policy. It uses no model training. The deployable policy is greedy max expected information gain under a uniform posterior over surviving verifier candidates. The oracle policy is target-aware and is included only as non-deployable headroom.

## Key Checks

- With four visible observations and budget 3, compare-gate greedy accuracy is 3.8% versus 73.8% for the target-aware oracle.
- Keeping four visible observations but raising active budget to 10, compare-gate greedy accuracy is 86.2% versus 98.8% for the target-aware oracle.
- Raising initial visible observations to sixteen while keeping budget 3 gives compare-gate greedy accuracy 91.2% versus 97.5% for the target-aware oracle.

## Interpretation

The greedy policy is the deployable one-step Bayesian experiment-design rule for a uniform posterior over all candidates consistent with observed executions. If additional budget or additional visible observations lift greedy performance, the bottleneck is information volume. If the target-aware oracle remains far above greedy, that gap should be read as target-knowledge headroom rather than directly recoverable deployable headroom.

## Budget-3 Summary

| Template | Visible observations | Greedy | Target-aware oracle | Gap points | Greedy candidates left | Greedy hidden-equivalent left |
|---|---:|---:|---:|---:|---:|---:|
| pair_affine_mod | 4 | 93.8% | 100.0% | 6.2 | 4.6 | 4.4 |
| pair_affine_mod | 8 | 100.0% | 100.0% | 0.0 | 4.3 | 4.3 |
| pair_affine_mod | 12 | 100.0% | 100.0% | 0.0 | 4.3 | 4.3 |
| pair_affine_mod | 16 | 100.0% | 100.0% | 0.0 | 4.3 | 4.3 |
| pair_compare_gate | 4 | 3.8% | 73.8% | 70.0 | 2005.4 | 293.8 |
| pair_compare_gate | 8 | 46.2% | 93.8% | 47.5 | 1162.6 | 263.0 |
| pair_compare_gate | 12 | 73.8% | 95.0% | 21.2 | 1256.9 | 248.7 |
| pair_compare_gate | 16 | 91.2% | 97.5% | 6.2 | 314.1 | 234.2 |

## Budget-10 Summary

| Template | Visible observations | Greedy | Target-aware oracle | Gap points | Greedy candidates left | Greedy hidden-equivalent left |
|---|---:|---:|---:|---:|---:|---:|
| pair_affine_mod | 4 | 100.0% | 100.0% | 0.0 | 4.3 | 4.3 |
| pair_affine_mod | 8 | 100.0% | 100.0% | 0.0 | 4.3 | 4.3 |
| pair_affine_mod | 12 | 100.0% | 100.0% | 0.0 | 4.3 | 4.3 |
| pair_affine_mod | 16 | 100.0% | 100.0% | 0.0 | 4.3 | 4.3 |
| pair_compare_gate | 4 | 86.2% | 98.8% | 12.5 | 707.4 | 244.8 |
| pair_compare_gate | 8 | 95.0% | 98.8% | 3.8 | 760.6 | 229.3 |
| pair_compare_gate | 12 | 96.2% | 98.8% | 2.5 | 529.1 | 222.8 |
| pair_compare_gate | 16 | 98.8% | 98.8% | 0.0 | 246.4 | 213.2 |

## Figures

- `reports/figures/budget_curve_pair_affine_mod.png`
- `reports/figures/budget_curve_pair_compare_gate.png`
- `reports/figures/oracle_gap_pair_affine_mod.png`
- `reports/figures/oracle_gap_pair_compare_gate.png`

## Reproduction

```bash
python scripts/eval_information_sweep.py --max-budget 10 --visible-extra 0 4 8 12
python scripts/make_report.py
```
