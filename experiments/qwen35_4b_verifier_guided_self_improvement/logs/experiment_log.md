# Experiment Log: Qwen3.5-4B Verifier-Guided Self-Improvement

Date: 2026-06-25

Experiment directory:
`/workspace/experiments/qwen35_4b_verifier_guided_self_improvement`

Large artifacts directory:
`/workspace/large_artifacts/qwen35_4b_verifier_guided_self_improvement`

## Objective

Test whether execution-verified self-training raises generation coverage on held-out coding tasks. The measurement is coverage x selection-capture: coverage asks whether any sampled candidate passes hidden tests, while selection-capture asks whether a deployable selector commits a hidden-correct candidate when one exists.

## Protocol

The loop is:

1. Sample candidates from Qwen3.5-4B.
2. Execute candidates against visible tests.
3. Fine-tune the generator on visible-test-passing candidates.
4. Re-sample with the updated generator.
5. Evaluate coverage, deployable pass@1, selection capture, and diversity on train, held-out, and transfer splits.

Hidden tests are used only for evaluation and oracle diagnostic ceilings, not for selecting self-training examples.

## Initial Plan

- Smoke gate: run a small sample on 10 train tasks, 20 held-out tasks, and 20 transfer tasks to verify execution, parsing, adapter loading, and SFT.
- Main gate: run MBPP train self-improvement and evaluate on at least 150 held-out MBPP test tasks plus 150 HumanEval transfer tasks if the smoke results and compute budget are viable.
- Controls: unverified self-training, oracle/reference SFT, and more-sampling matched-compute baseline.

## Running Notes

- Created standalone experiment directory and separate large artifact directory.
- Added package-local Qwen loading and Python execution utilities.
- Added `sample_round.py` for round-specific sampling with optional generator LoRA adapter.
- Added `build_generator_sft.py` for verified, unverified, and oracle/reference SFT datasets.
- Added `train_generator_sft.py` for QLoRA generator fine-tuning.

## Smoke Gate

Smoke settings:

- MBPP train: 10 tasks.
- MBPP held-out: 20 tasks.
- Direct samples/task: 3.
- Repair attempts/task: 1.
- Max new tokens: 180.
- Verified-SFT steps: 30.

Smoke train sampling completed:

- Mean candidates/task: 3.2.
- Hidden coverage: 50.0%.
- Visible coverage: 50.0%.
- Mean visible-passers/task: 1.7.

The visible-test filter produced 12 smoke SFT examples from 10 tasks; 10 of those also passed hidden tests.

Smoke verified-SFT adapter trained successfully and could be loaded for generation.

Smoke held-out comparison on 20 MBPP test tasks:

| Arm | Coverage | Visible coverage | Mean candidates | Mean visible-passers |
|---|---:|---:|---:|---:|
| base | 65.0% | 65.0% | 2.95 | 1.80 |
| selftrain_verified_r1_smoke | 70.0% | 70.0% | 2.65 | 1.55 |

Decision: proceed to a larger run. To keep the run tractable while preserving the load-bearing held-out size, the main config is tightened to 80 MBPP train tasks, 150 MBPP held-out tasks, 150 HumanEval transfer tasks, 4 direct samples/task, 1 repair attempt/task, and 80 generator-SFT steps.

## Main Run Ledger

Main base sampling on 80 MBPP train tasks completed:

- Mean candidates/task: 3.74.
- Hidden coverage: 70.0%.
- Visible coverage: 70.0%.
- Mean visible-passers/task: 2.08.

SFT dataset construction from the base train pool:

| Dataset | Examples | Hidden-positive examples | Visible-positive examples |
|---|---:|---:|---:|
| verified | 115 | 105 | 115 |
| unverified | 158 | 95 | 103 |
| oracle/reference | 80 | 80 | 80 |

The `selftrain_verified_r1` adapter was trained for 80 steps on the verified examples.

Main held-out and transfer sampling results:

| Split | Arm | Coverage | Visible coverage | Mean candidates | Mean visible-passers |
|---|---|---:|---:|---:|---:|
| MBPP train | base | 70.0% | 70.0% | 3.74 | 2.08 |
| MBPP train | selftrain_verified_r1 | 71.2% | 71.2% | 2.83 | 1.76 |
| MBPP held-out | base | 65.3% | 65.3% | 3.71 | 1.98 |
| MBPP held-out | selftrain_verified_r1 | 64.7% | 64.7% | 3.40 | 1.71 |
| HumanEval transfer | base | 75.3% | 75.3% | 3.08 | 2.54 |
| HumanEval transfer | selftrain_verified_r1 | 74.7% | 74.7% | 2.77 | 2.35 |

Decision after round 1: do not run rounds 2 and 3. The pre-registered gate required held-out coverage to move in the right direction; the 150-task held-out readout regressed slightly despite a small train increase. The smoke +5 point signal did not replicate at larger n.

The unverified self-training control was trained for 80 steps on the unfiltered parsed/safe sample set. Held-out MBPP sampling completed:

- Hidden coverage: 62.0%.
- Visible coverage: 62.0%.
- Mean candidates/task: 3.01.
- Mean visible-passers/task: 1.63.

The oracle/reference SFT control was trained for 80 steps on reference solutions for the same 80 MBPP train tasks. Held-out MBPP sampling completed:

- Hidden coverage: 63.3%.
- Visible coverage: 63.3%.
- Mean candidates/task: 3.83.
- Mean visible-passers/task: 2.00.

Interim interpretation: verification filtering is load-bearing relative to unverified self-training, but under this data and LoRA budget it does not beat the base generator on held-out MBPP or HumanEval transfer. Oracle/reference SFT also does not beat base on held-out MBPP, which points to a broader small-SFT/generalization limit rather than only noisy self-generated labels.

The matched-compute sample-more baseline used the base generator with 8 direct samples/task and 2 repair attempts/task on the same 150 MBPP held-out tasks. It completed in 59:56 wall time.

- Hidden coverage: 68.7%.
- Visible coverage: 68.7%.
- Mean candidates/task: 7.01.
- Mean visible-passers/task: 3.60.

This baseline beat base coverage (65.3%) and all fine-tuned generator arms, but did so at roughly double the sampled candidate count and a much higher execution/runtime cost. This is the strongest practical result: preserving and increasing sampling diversity is a better use of the available compute than the tested 80-step SFT loops.

## Final Artifacts and Audit

Final report generated:

- `reports/qwen35_4b_verifier_guided_self_improvement_report.md`
- `reports/summary_coverage.csv`
- `reports/summary_commit.csv`
- `reports/figures/coverage_by_arm.png`
- `reports/figures/diversity_and_pool_size.png`
- `reports/figures/commit_accuracy_heldout.png`
- `reports/figures/training_losses.png`

Audit results:

- All package scripts compile with `python -m py_compile src/*.py scripts/*.py`.
- The experiment package contains no files larger than 10 MB.
- The experiment package is 18 MB.
- Large LoRA/tokenizer artifacts are isolated under `/workspace/large_artifacts/qwen35_4b_verifier_guided_self_improvement` (401 MB).
- The report, log, config, source, and scripts are standalone and do not refer to earlier experiments.
