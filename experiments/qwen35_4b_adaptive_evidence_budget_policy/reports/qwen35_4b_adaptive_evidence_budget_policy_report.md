# Qwen3.5-4B Adaptive Evidence Budget Policy

## Objective

This standalone experiment tests whether Qwen3.5-4B can be posttrained as a STOP/MORE controller for an executable verifier. The verifier supplies the best deployable next probe by target-independent expected split; the model decides whether to commit now or spend another probe, up to a maximum budget of ten.

## Data

- Train records: 240.
- Eval records: 160.
- Train STOP/MORE states: 2640.
- Eval STOP/MORE states: 1760.

## Key Result

- SFT STOP/MORE reached 92.5% accuracy using 4.81 probes on average.
- Fixed budget 3/6/10 reached 45.0% / 74.4% / 92.5%.
- Oracle stopping reached 92.5% using 4.22 probes on average.

## Overall

| Policy | Hidden-all accuracy | Avg probes | Candidates left | Hidden-equivalent left |
|---|---:|---:|---:|---:|
| Fixed budget 3 | 45.0% | 3.00 | 1040.5 | 12.9 |
| Fixed budget 6 | 74.4% | 6.00 | 544.7 | 8.8 |
| Fixed budget 10 | 92.5% | 10.00 | 353.8 | 3.6 |
| Threshold <=100 | 6.9% | 1.86 | 397.6 | 4.3 |
| Threshold <=1000 | 5.0% | 0.94 | 600.7 | 4.8 |
| Oracle stop | 92.5% | 4.22 | 356.2 | 3.8 |
| Base Qwen stop/more | 5.0% | 0.04 | 3819.4 | 129.6 |
| SFT Qwen stop/more | 92.5% | 4.81 | 353.8 | 3.6 |

## By Template

Cells show `accuracy / average probes`.

| Policy | Affine-mod | Compare-gate |
|---|---:|---:|
| Fixed budget 3 | 87.5% / 3.00 | 2.5% / 3.00 |
| Fixed budget 6 | 98.8% / 6.00 | 50.0% / 6.00 |
| Fixed budget 10 | 98.8% / 10.00 | 86.2% / 10.00 |
| Threshold <=100 | 12.5% / 0.07 | 1.2% / 3.65 |
| Threshold <=1000 | 10.0% / 0.00 | 0.0% / 1.88 |
| Oracle stop | 98.8% / 1.82 | 86.2% / 6.62 |
| Base Qwen stop/more | 10.0% / 0.00 | 0.0% / 0.09 |
| SFT Qwen stop/more | 98.8% / 2.34 | 86.2% / 7.28 |

## Interpretation

This test separates the value of the adaptive inference loop from the value of learning the stop rule. If the SFT policy lies on or above the fixed-budget Pareto curve, posttraining learned useful budget control. If fixed budgets dominate it, the practical lever is simply allowing more executable observations and using a transparent budget rule.

## Figures

- `reports/figures/accuracy_vs_probes.png`
- `reports/figures/accuracy_by_template.png`
- `reports/figures/probes_by_template.png`
- `reports/figures/budget_sft_loss.png`

## Reproduction

```bash
python scripts/build_dataset.py --train-per-cell 40 --eval-per-cell 20 --query-pool-cases 96 --max-budget 10
python scripts/eval_budget_policy.py --policy fixed --name fixed_budget3 --fixed-budget 3 --max-budget 10
python scripts/eval_budget_policy.py --policy fixed --name fixed_budget6 --fixed-budget 6 --max-budget 10
python scripts/eval_budget_policy.py --policy fixed --name fixed_budget10 --fixed-budget 10 --max-budget 10
python scripts/eval_budget_policy.py --policy threshold --name threshold_100 --threshold 100 --max-budget 10
python scripts/eval_budget_policy.py --policy threshold --name threshold_1000 --threshold 1000 --max-budget 10
python scripts/eval_budget_policy.py --policy oracle_stop --name oracle_stop --max-budget 10
python scripts/eval_budget_policy.py --policy base --name base_budget_policy --max-budget 10
python scripts/train_budget_sft.py --max-steps 220 --batch-size 2 --grad-accum 2
python scripts/eval_budget_policy.py --policy adapter --name sft_budget_policy --adapter-dir /workspace/large_artifacts/qwen35_4b_adaptive_evidence_budget_policy/models/budget_sft_lora --max-budget 10
python scripts/make_report.py
```
