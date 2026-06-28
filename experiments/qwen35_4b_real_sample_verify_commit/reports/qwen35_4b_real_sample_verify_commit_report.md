# Qwen3.5-4B Real Sample Verify Commit

## Objective

Run the full sample -> verify -> commit loop using genuine Qwen3.5-4B code samples. The primary measurement decomposes final pass rate into candidate-pool coverage and selector capture.

## Candidate Generation

- MBPP train records: 40.
- MBPP eval records: 20.
- HumanEval eval records: 30.
- Direct samples per task: 8.
- Model repair attempts per task: 1.
- Temperatures: 0.2,0.7,1.0.
- Max new tokens: 220.

No mutation-generated candidates are used. Every candidate is either a direct Qwen sample or a Qwen repair sample.

## Key Result

- MBPP eval coverage at max budget: 60.0%.
- MBPP eval first-visible commit: 55.0%.
- MBPP eval SFT verifier commit: 60.0%.
- HumanEval coverage at max budget: 96.7%.
- HumanEval first-visible commit: 96.7%.
- HumanEval SFT verifier commit: 96.7%.
- MBPP SFT stop controller: 55.0% using 4.20 samples on average.
- HumanEval SFT stop controller: 96.7% using 1.70 samples on average.

## Readout

On these genuine sampled pools, selection is not the main bottleneck. HumanEval is already easy for the generator at this budget: coverage reaches 96.7%, and first-visible, frozen verifier, SFT verifier, and oracle selection all match the ceiling. MBPP is coverage-limited: max-budget coverage is 60.0%, and the verifier can capture that ceiling, but it cannot create missing correct programs. The adaptive controller mostly trades a small amount of MBPP accuracy for fewer samples, while matching the HumanEval ceiling with far fewer samples.

## Fixed-Budget Summary

| Dataset | Policy | Budget | Visible coverage | Selected hidden-pass | Coverage captured | Samples seen |
|---|---|---:|---:|---:|---:|---:|
| humaneval | base_verifier | 1 | 90.0% | 90.0% | 100.0% | 1.00 |
| humaneval | base_verifier | 2 | 90.0% | 90.0% | 100.0% | 1.97 |
| humaneval | base_verifier | 4 | 93.3% | 93.3% | 100.0% | 3.73 |
| humaneval | base_verifier | 8 | 96.7% | 96.7% | 100.0% | 5.40 |
| humaneval | base_verifier | 9 | 96.7% | 96.7% | 100.0% | 5.47 |
| humaneval | first_visible | 1 | 90.0% | 90.0% | 100.0% | 1.00 |
| humaneval | first_visible | 2 | 90.0% | 90.0% | 100.0% | 1.97 |
| humaneval | first_visible | 4 | 93.3% | 93.3% | 100.0% | 3.73 |
| humaneval | first_visible | 8 | 96.7% | 96.7% | 100.0% | 5.40 |
| humaneval | first_visible | 9 | 96.7% | 96.7% | 100.0% | 5.47 |
| humaneval | oracle_coverage | 1 | 90.0% | 90.0% | 100.0% | 1.00 |
| humaneval | oracle_coverage | 2 | 90.0% | 90.0% | 100.0% | 1.97 |
| humaneval | oracle_coverage | 4 | 93.3% | 93.3% | 100.0% | 3.73 |
| humaneval | oracle_coverage | 8 | 96.7% | 96.7% | 100.0% | 5.40 |
| humaneval | oracle_coverage | 9 | 96.7% | 96.7% | 100.0% | 5.47 |
| humaneval | public_signature_majority | 1 | 90.0% | 90.0% | 100.0% | 1.00 |
| humaneval | public_signature_majority | 2 | 90.0% | 90.0% | 100.0% | 1.97 |
| humaneval | public_signature_majority | 4 | 93.3% | 93.3% | 100.0% | 3.73 |
| humaneval | public_signature_majority | 8 | 96.7% | 96.7% | 100.0% | 5.40 |
| humaneval | public_signature_majority | 9 | 96.7% | 96.7% | 100.0% | 5.47 |
| humaneval | sft_verifier | 1 | 90.0% | 90.0% | 100.0% | 1.00 |
| humaneval | sft_verifier | 2 | 90.0% | 90.0% | 100.0% | 1.97 |
| humaneval | sft_verifier | 4 | 93.3% | 93.3% | 100.0% | 3.73 |
| humaneval | sft_verifier | 8 | 96.7% | 96.7% | 100.0% | 5.40 |
| humaneval | sft_verifier | 9 | 96.7% | 96.7% | 100.0% | 5.47 |
| humaneval | shortest_visible | 1 | 90.0% | 90.0% | 100.0% | 1.00 |
| humaneval | shortest_visible | 2 | 90.0% | 90.0% | 100.0% | 1.97 |
| humaneval | shortest_visible | 4 | 93.3% | 93.3% | 100.0% | 3.73 |
| humaneval | shortest_visible | 8 | 96.7% | 96.7% | 100.0% | 5.40 |
| humaneval | shortest_visible | 9 | 96.7% | 96.7% | 100.0% | 5.47 |
| mbpp | base_verifier | 1 | 35.0% | 35.0% | 100.0% | 1.00 |
| mbpp | base_verifier | 2 | 35.0% | 35.0% | 100.0% | 2.00 |
| mbpp | base_verifier | 4 | 50.0% | 50.0% | 100.0% | 4.00 |
| mbpp | base_verifier | 8 | 60.0% | 60.0% | 100.0% | 7.45 |
| mbpp | base_verifier | 9 | 60.0% | 60.0% | 100.0% | 7.80 |
| mbpp | first_visible | 1 | 35.0% | 35.0% | 100.0% | 1.00 |
| mbpp | first_visible | 2 | 35.0% | 35.0% | 100.0% | 2.00 |
| mbpp | first_visible | 4 | 50.0% | 50.0% | 100.0% | 4.00 |
| mbpp | first_visible | 8 | 60.0% | 55.0% | 91.7% | 7.45 |
| mbpp | first_visible | 9 | 60.0% | 55.0% | 91.7% | 7.80 |
| mbpp | oracle_coverage | 1 | 35.0% | 35.0% | 100.0% | 1.00 |
| mbpp | oracle_coverage | 2 | 35.0% | 35.0% | 100.0% | 2.00 |
| mbpp | oracle_coverage | 4 | 50.0% | 50.0% | 100.0% | 4.00 |
| mbpp | oracle_coverage | 8 | 60.0% | 60.0% | 100.0% | 7.45 |
| mbpp | oracle_coverage | 9 | 60.0% | 60.0% | 100.0% | 7.80 |
| mbpp | public_signature_majority | 1 | 35.0% | 35.0% | 100.0% | 1.00 |
| mbpp | public_signature_majority | 2 | 35.0% | 35.0% | 100.0% | 2.00 |
| mbpp | public_signature_majority | 4 | 50.0% | 50.0% | 100.0% | 4.00 |
| mbpp | public_signature_majority | 8 | 60.0% | 55.0% | 91.7% | 7.45 |
| mbpp | public_signature_majority | 9 | 60.0% | 55.0% | 91.7% | 7.80 |
| mbpp | sft_verifier | 1 | 35.0% | 35.0% | 100.0% | 1.00 |
| mbpp | sft_verifier | 2 | 35.0% | 35.0% | 100.0% | 2.00 |
| mbpp | sft_verifier | 4 | 50.0% | 50.0% | 100.0% | 4.00 |
| mbpp | sft_verifier | 8 | 60.0% | 60.0% | 100.0% | 7.45 |
| mbpp | sft_verifier | 9 | 60.0% | 60.0% | 100.0% | 7.80 |
| mbpp | shortest_visible | 1 | 35.0% | 35.0% | 100.0% | 1.00 |
| mbpp | shortest_visible | 2 | 35.0% | 35.0% | 100.0% | 2.00 |
| mbpp | shortest_visible | 4 | 50.0% | 50.0% | 100.0% | 4.00 |
| mbpp | shortest_visible | 8 | 60.0% | 60.0% | 100.0% | 7.45 |
| mbpp | shortest_visible | 9 | 60.0% | 60.0% | 100.0% | 7.80 |

## Adaptive-Budget Summary

| Dataset | Policy | Visible coverage | Selected hidden-pass | Coverage captured | Mean samples used |
|---|---|---:|---:|---:|---:|
| humaneval | oracle_stop | 96.7% | 96.7% | 100.0% | 1.60 |
| humaneval | sft_stop_controller | 96.7% | 96.7% | 100.0% | 1.70 |
| humaneval | threshold_sft_score | 96.7% | 96.7% | 100.0% | 1.60 |
| mbpp | oracle_stop | 60.0% | 60.0% | 100.0% | 5.35 |
| mbpp | sft_stop_controller | 55.0% | 55.0% | 100.0% | 4.20 |
| mbpp | threshold_sft_score | 55.0% | 55.0% | 100.0% | 4.20 |

## Interpretation

This experiment uses genuinely sampled candidate pools, so the central fork is visible directly: under this sampling setup, the verifier is not failing on subtle visible-passing near-misses at this scale. The dominant open problem is generating a correct candidate on harder MBPP tasks. For HumanEval, the first visible-passing sample is usually already correct, so verifier posttraining and adaptive control add little accuracy headroom, though adaptive stopping reduces sample cost.

The main limitation is scale: this run uses 20 MBPP eval tasks and 30 HumanEval tasks because genuine Qwen sampling is the expensive step. The conclusion should be read as a measured pilot of the real candidate distribution, not as a final benchmark score.

The next high-leverage step is generator-side: increase coverage through better sampling, repair, or verifier-guided self-improvement. Verifier work should focus on larger, more adversarial sampled pools, because this run did not surface a meaningful selector wall.

## Figures

- `reports/figures/mbpp_coverage_curve.png`
- `reports/figures/mbpp_fixed_budget_accuracy.png`
- `reports/figures/mbpp_adaptive_budget.png`
- `reports/figures/humaneval_coverage_curve.png`
- `reports/figures/humaneval_fixed_budget_accuracy.png`
- `reports/figures/humaneval_adaptive_budget.png`

## Reproduction

```bash
python scripts/sample_candidates.py --mbpp-train 40 --mbpp-eval 20 --humaneval-eval 30 --samples-per-task 8 --repair-per-task 1 --max-new-tokens 220 --generation-batch-size 4 --temperatures 0.2,0.7,1.0 --top-p 0.95
python scripts/build_verifier_examples.py
python scripts/train_action_sft.py --train data/train_verifier_examples.jsonl --output-dir /workspace/large_artifacts/qwen35_4b_real_sample_verify_commit/models/verifier_sft_lora --loss-out reports/verifier_sft_training_losses.json --method sampled_semantic_verifier_sft --max-steps 160 --batch-size 2 --grad-accum 2
python scripts/eval_commit.py --records data/mbpp_train_records.jsonl --policy sft_verifier --name sft_verifier --out reports/eval/mbpp_train_sft_verifier.json --budgets 1,2,4,8,max --adapter-dir /workspace/large_artifacts/qwen35_4b_real_sample_verify_commit/models/verifier_sft_lora
python scripts/build_stop_examples.py --scores reports/eval/mbpp_train_sft_verifier.json --budgets 1,2,4,8,max
python scripts/train_action_sft.py --train data/train_stop_examples.jsonl --output-dir /workspace/large_artifacts/qwen35_4b_real_sample_verify_commit/models/stop_sft_lora --loss-out reports/stop_sft_training_losses.json --method sampled_generation_budget_stop_sft --max-steps 120 --batch-size 2 --grad-accum 2
bash scripts/run_evaluation_suite.sh
python scripts/make_report.py
```
