# Qwen3.5-4B Verifier-Guided Self-Improvement Report

Date: 2026-06-25

## Executive Read

The main result is negative for the central question. Verified self-training did not raise held-out generation coverage under this local LoRA/data budget. The 20-task smoke signal was positive, but the 150-task held-out run regressed slightly.

- MBPP held-out: 65.3% -> 64.7% (-0.7 pp).
- HumanEval transfer: 75.3% -> 74.7% (-0.7 pp).
- MBPP train: 70.0% -> 71.2% (+1.3 pp).

Rounds 2 and 3 were intentionally stopped at the pre-registered gate because held-out coverage did not move in the right direction after round 1.

The controls sharpen the read:

- Unverified self-training is worse than verified self-training on MBPP held-out, so the execution filter is load-bearing.
- Oracle/reference SFT on the same 80 train tasks also does not beat base on MBPP held-out, so the failure is not only noisy self-generated labels.
- More inference sampling beats the training arms on MBPP held-out: 65.3% -> 68.7%.

## Coverage

| Split    | Arm                         | n   | Coverage | Visible cov. | Candidates | Distinct |
| -------- | --------------------------- | --- | -------- | ------------ | ---------- | -------- |
| train    | base                        | 80  | 70.0%    | 70.0%        | 3.74       | 96.2%    |
| train    | selftrain_verified_r1       | 80  | 71.2%    | 71.2%        | 2.83       | 98.4%    |
| heldout  | base                        | 150 | 65.3%    | 65.3%        | 3.71       | 97.3%    |
| heldout  | selftrain_verified_r1       | 150 | 64.7%    | 64.7%        | 3.40       | 97.9%    |
| heldout  | selftrain_unverified        | 150 | 62.0%    | 62.0%        | 3.01       | 98.3%    |
| heldout  | oracle_sft                  | 150 | 63.3%    | 63.3%        | 3.83       | 99.8%    |
| heldout  | sample_more_matched_compute | 150 | 68.7%    | 68.7%        | 7.01       | 97.5%    |
| transfer | base                        | 150 | 75.3%    | 75.3%        | 3.08       | 91.7%    |
| transfer | selftrain_verified_r1       | 150 | 74.7%    | 74.7%        | 2.77       | 93.9%    |

Primary metric: coverage is pass@K for the sampled pool, meaning at least one candidate passes hidden tests. Hidden tests were not used for self-training selection.

## Commit Selection

| Arm                         | Policy                    | Budget | Selected | Coverage captured |
| --------------------------- | ------------------------- | ------ | -------- | ----------------- |
| base                        | first_visible             | 5      | 63.3%    | 96.9%             |
| base                        | public_signature_majority | 5      | 63.3%    | 96.9%             |
| base                        | base_verifier             | 5      | 64.7%    | 99.0%             |
| base                        | oracle_coverage           | 5      | 65.3%    | 100.0%            |
| selftrain_verified_r1       | first_visible             | 5      | 61.3%    | 94.8%             |
| selftrain_verified_r1       | public_signature_majority | 5      | 61.3%    | 94.8%             |
| selftrain_verified_r1       | base_verifier             | 5      | 63.3%    | 97.9%             |
| selftrain_verified_r1       | oracle_coverage           | 5      | 64.7%    | 100.0%            |
| selftrain_unverified        | first_visible             | 5      | 60.0%    | 96.8%             |
| selftrain_unverified        | public_signature_majority | 5      | 60.0%    | 96.8%             |
| selftrain_unverified        | base_verifier             | 5      | 62.0%    | 100.0%            |
| selftrain_unverified        | oracle_coverage           | 5      | 62.0%    | 100.0%            |
| oracle_sft                  | first_visible             | 5      | 60.7%    | 95.8%             |
| oracle_sft                  | public_signature_majority | 5      | 60.7%    | 95.8%             |
| oracle_sft                  | base_verifier             | 5      | 61.3%    | 96.8%             |
| oracle_sft                  | oracle_coverage           | 5      | 63.3%    | 100.0%            |
| sample_more_matched_compute | first_visible             | 10     | 66.7%    | 97.1%             |
| sample_more_matched_compute | public_signature_majority | 10     | 66.7%    | 97.1%             |
| sample_more_matched_compute | base_verifier             | 10     | 66.7%    | 97.1%             |
| sample_more_matched_compute | oracle_coverage           | 10     | 68.7%    | 100.0%            |

Selection remains secondary here because coverage is the binding quantity. `oracle_coverage` is the diagnostic upper bound: if a hidden-correct candidate exists in the pool, it commits one.

## Figures

- [Coverage by arm](figures/coverage_by_arm.png)
- [Diversity and pool size](figures/diversity_and_pool_size.png)
- [Commit accuracy on held-out pools](figures/commit_accuracy_heldout.png)
- [Generator SFT losses](figures/training_losses.png)

## Interpretation

This run does not support the hypothesis that one round of verified rejection-sampling SFT expands Qwen3.5-4B's coding frontier on held-out tasks. It mostly narrows the pool: candidate count, visible-passers, and transfer coverage all decrease slightly after verified SFT. The best current deployable lever in this package is not small-SFT self-improvement; it is preserving or increasing sample diversity and then using execution/selection to harvest coverage.

A stronger future positive would need to change at least one of these constraints: substantially more train tasks, stronger multi-round data accumulation without diversity collapse, curriculumed repair data for tasks with zero initial coverage, or a generator objective that explicitly preserves pass@K diversity rather than only imitating passing samples.

## Artifacts

- Experiment package: `/workspace/experiments/qwen35_4b_verifier_guided_self_improvement`
- Large adapters/checkpoints: `/workspace/large_artifacts/qwen35_4b_verifier_guided_self_improvement`
- Coverage CSV: `reports/summary_coverage.csv`
- Commit CSV: `reports/summary_commit.csv`
