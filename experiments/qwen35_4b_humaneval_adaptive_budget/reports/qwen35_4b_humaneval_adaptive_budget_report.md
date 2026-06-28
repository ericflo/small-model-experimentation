# Qwen3.5-4B HumanEval Adaptive Evidence Budget

## Objective

This standalone experiment tests whether Qwen3.5-4B can be posttrained as a STOP/MORE controller for an executable Python verifier on HumanEval tasks. The verifier generates candidate implementations, chooses unlabeled probes by target-independent output-agreement split, and commits the first candidate in the largest output-agreement cluster. The model only decides whether to commit or spend one more executable probe.

## Dataset

- Source: `openai/openai_humaneval`.
- Train tasks: 24; eval tasks: 12.
- Visible tests per task: 1; probe pool: 8; generated hidden tests: 0.
- Candidate implementations per task: 16 maximum, from canonical-solution mutations and generic fallback bodies.
- Public doctest examples are the only labeled visible tests. Generated probes are unlabeled and are used only to form candidate agreement clusters. Reference outputs for generated probes are stored only as audit metadata.
- STOP/MORE train states: 216; eval states: 108.

## Key Result

- SFT STOP/MORE reached 16.7% hidden-correct selection with 3.83 probes on average.
- Fixed budget 8 reached 16.7% with 8.00 probes.
- Oracle stopping reached 16.7% with 6.67 probes.
- Base Qwen STOP/MORE reached 16.7%.
- Candidate-pool coverage was 100.0%: a hidden-correct candidate was present for every eval task, but the leak-free agreement selector usually did not choose it.

## Overall

| Policy | Hidden-correct selected | Candidate-pool coverage | Avg probes | Agreement clusters | Selected-cluster share | Hidden-correct survivors |
|---|---:|---:|---:|---:|---:|---:|
| Fixed budget 0 | 16.7% | 100.0% | 0.00 | 1.00 | 100.0% | 1.33 |
| Fixed budget 3 | 16.7% | 100.0% | 3.00 | 2.92 | 56.6% | 1.33 |
| Fixed budget 6 | 16.7% | 100.0% | 6.00 | 2.92 | 56.6% | 1.33 |
| Fixed budget 8 | 16.7% | 100.0% | 8.00 | 2.92 | 56.6% | 1.33 |
| Stop if cluster >=70% | 16.7% | 100.0% | 0.00 | 1.00 | 100.0% | 1.33 |
| Stop if cluster >=90% | 16.7% | 100.0% | 0.00 | 1.00 | 100.0% | 1.33 |
| Oracle stop | 16.7% | 100.0% | 6.67 | 2.92 | 56.6% | 1.33 |
| Base Qwen stop/more | 16.7% | 100.0% | 0.33 | 1.75 | 84.1% | 1.33 |
| SFT Qwen stop/more | 16.7% | 100.0% | 3.83 | 2.08 | 69.8% | 1.33 |

## Interpretation

This pilot is a clean negative for STOP/MORE budget control under this leak-free HumanEval evidence model. The candidate pool contains a hidden-correct implementation for every eval task, so the low final accuracy is not a coverage failure. The failure is that unlabeled generated probes split the visible-passing candidates into agreement clusters without grounding which cluster is correct. Spending more probes changes cluster structure but does not move the selected candidate onto the hidden-correct implementation, so even a hidden-aware oracle stopping rule has no accuracy to recover.

The SFT controller learned a cheaper stopping behavior than fixed budget 8, but because the selected candidate remained wrong on 11 of 12 eval tasks, the cost saving is not useful. For this benchmark shape, the next useful lever is not a better STOP/MORE controller over unlabeled tests; it is either a stronger deployable candidate selector, a trustworthy labeled-test generator, or an adaptive generation budget that samples more candidate programs when the visible-passing pool is poorly grounded.

This is a pilot-scale result. HumanEval public examples and randomly generated in-domain probes limited the usable split to 24 train and 12 eval tasks. The conclusion should be read as a substrate diagnostic, not as a benchmark-level pass-rate estimate.

## Figures

- `reports/figures/accuracy_vs_probes.png`
- `reports/figures/accuracy_by_policy.png`
- `reports/figures/probes_by_policy.png`
- `reports/figures/budget_sft_loss.png`

## Reproduction

```bash
python scripts/build_dataset.py --train-tasks 24 --eval-tasks 12 --visible-tests 1 --probe-tests 8 --hidden-tests 0 --candidate-count 16 --max-budget 8
python scripts/eval_budget_policy.py --policy fixed --name fixed_budget0 --fixed-budget 0 --max-budget 8
python scripts/eval_budget_policy.py --policy fixed --name fixed_budget3 --fixed-budget 3 --max-budget 8
python scripts/eval_budget_policy.py --policy fixed --name fixed_budget6 --fixed-budget 6 --max-budget 8
python scripts/eval_budget_policy.py --policy fixed --name fixed_budget8 --fixed-budget 8 --max-budget 8
python scripts/eval_budget_policy.py --policy threshold --name threshold_70 --threshold 70 --max-budget 8
python scripts/eval_budget_policy.py --policy threshold --name threshold_90 --threshold 90 --max-budget 8
python scripts/eval_budget_policy.py --policy oracle_stop --name oracle_stop --max-budget 8
python scripts/eval_budget_policy.py --policy base --name base_budget_policy --max-budget 8
python scripts/train_budget_sft.py --max-steps 160 --batch-size 2 --grad-accum 2
python scripts/eval_budget_policy.py --policy adapter --name sft_budget_policy --adapter-dir /workspace/large_artifacts/qwen35_4b_humaneval_adaptive_budget/models/budget_sft_lora --max-budget 8
python scripts/make_report.py
```
