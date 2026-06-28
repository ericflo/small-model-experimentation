# Experiment Log: Qwen3.5-4B Trained vs Frozen Repair MDP

Date: 2026-06-25

Experiment directory:
`/workspace/experiments/qwen35_4b_trained_vs_frozen_repair_mdp`

Large artifacts directory:
`/workspace/large_artifacts/qwen35_4b_trained_vs_frozen_repair_mdp`

## Objective

Test whether a trained repair policy expands held-out generation coverage beyond frozen Qwen self-repair at honestly matched model-forward-token budget.

The headline metric is zero-to-one hidden coverage lift: among held-out tasks where the direct sample pool contains no hidden-correct candidate, how many become covered after repair.

## Design Commitments

- The frozen Qwen repair loop is the primary baseline.
- The sample-more baseline is matched by estimated forward tokens, not by candidate count.
- Training labels may use full train-task tests, but repair prompts contain only task text, public tests, candidate code, and visible execution traces.
- Hidden tests are used for evaluation and train-side labeling only; they are never included in repair prompts.
- False repair rate is tracked: visible-pass repair candidates that fail hidden tests.
- Diversity is tracked with behavior signatures after each arm.
- Rounds beyond the smoke gate are skipped if the pre-registered gates fail.

## Initial Package

- Created package-local source utilities, direct sampler, repair rollout runner, repair dataset builder, SFT trainer, DPO trainer, token-matched sample-more baseline, and commit evaluator.
- Large LoRA artifacts will be stored outside the experiment directory.

## Smoke Gate

Smoke direct sampling on 12 MBPP train tasks completed:

- Hidden coverage: 50.0%.
- Mean candidates/task: 2.92.
- Mean hidden-pass candidates/task: 1.33.
- Estimated forward tokens: 8,525.

Frozen repair on the same smoke train pool completed:

- Hidden coverage remained 50.0%.
- Zero-base records: 6.
- Zero-to-one repairs: 0.
- Visible-passing repair candidates: 2.
- False repairs among visible-passing repairs: 1.
- Estimated repair forward tokens: 6,020.

The smoke repair dataset had only 1 SFT example and 1 DPO pair, so the SFT gate was not met. Decision: expand train repair mining before training rather than fitting a meaningless adapter.

## Train Repair Mining

Main direct sampling on 100 MBPP train tasks completed:

- Hidden coverage: 72.0%.
- Zero-coverage train tasks: 28.
- Mean candidates/task: 3.42.
- Mean hidden-pass candidates/task: 1.89.
- Estimated forward tokens: 93,141.

One-attempt frozen repair mining on this train pool completed:

- Hidden coverage: 75.0%.
- Zero-to-one repairs: 3 / 28.
- Visible-passing repair candidates: 15.
- False repair rate among visible-passing repairs: 13.3%.
- Estimated repair forward tokens: 50,830.
- Dataset: 13 SFT examples, 13 DPO pairs.

Expanded train-only repair mining with more sources and two attempts/source completed:

- Hidden coverage: 76.0%.
- Zero-to-one repairs: 4 / 28.
- Visible-passing repair candidates: 26.
- False repair rate among visible-passing repairs: 30.8%.
- Estimated repair forward tokens: 136,948.
- Dataset: 17 SFT examples, 15 DPO pairs.

Decision: train the repair SFT adapter on the expanded mined set. The signal is thin, so the trained-arm interpretation must be conservative and judged primarily against frozen repair.

## Repair SFT

Repair SFT completed on the expanded mined training set:

- Train examples: 17.
- Max steps: 80.
- Batch size / grad accumulation: 1 / 4.
- Learning rate: 1e-4.
- Max sequence length: 1,536.
- Final logged loss: 0.0135.
- Adapter directory: `/workspace/large_artifacts/qwen35_4b_trained_vs_frozen_repair_mdp/models/repair_sft_lora`.

Interpretation note: this adapter fit a very small, mined repair set. The loss confirms the optimizer can fit the examples, but the experiment's useful readout is whether the adapter beats frozen repair on held-out zero-to-one coverage without increasing false repairs.

## Held-Out Direct Pool

Direct sampling on 150 MBPP held-out tasks completed:

- Hidden coverage: 62.0% (93 / 150 tasks).
- Zero-coverage held-out denominator: 57 / 150 tasks.
- Visible coverage: 62.0%.
- Mean candidates/task: 3.45.
- Mean hidden-pass candidates/task: 1.63.
- Mean visible-pass candidates/task: 1.96.
- Mean distinct behavior rate: 0.77.
- Estimated forward tokens: 138,902.

This locks the headline denominator. Repair arms are judged by how many of the 57 zero-coverage tasks become hidden-covered, and by whether they do so beyond frozen repair at comparable model-forward-token cost.

## Held-Out Frozen Repair

Frozen Qwen repair on the held-out direct pool completed:

- Hidden coverage: 64.0% (96 / 150 tasks).
- Zero-to-one repairs: 3 / 57.
- Zero-to-one rate: 5.3%.
- Visible-passing repair candidates: 28.
- False repairs among visible-passing repairs: 7.
- False repair rate: 25.0%.
- Mean candidates/task: 4.01.
- Mean hidden-pass candidates/task: 1.77.
- Mean distinct behavior rate: 0.754.
- Estimated repair forward tokens: 79,614.

Interpretation note: frozen repair creates a small but real held-out frontier lift, but one quarter of visible-passing repairs fail hidden tests. A trained arm must improve the zero-to-one count without worsening this false-repair profile.

## Held-Out Repair SFT

SFT-adapter repair on the same held-out direct pool completed:

- Hidden coverage: 63.3% (95 / 150 tasks).
- Zero-to-one repairs: 2 / 57.
- Zero-to-one rate: 3.5%.
- Visible-passing repair candidates: 24.
- False repairs among visible-passing repairs: 7.
- False repair rate: 29.2%.
- Mean candidates/task: 3.98.
- Mean hidden-pass candidates/task: 1.75.
- Mean distinct behavior rate: 0.766.
- Estimated repair forward tokens: 79,325.

Gate decision: skip DPO. The SFT adapter underperformed frozen repair on the headline metric (2 vs 3 zero-to-one repairs) and had a worse false-repair rate (29.2% vs 25.0%). Running DPO from this checkpoint would add variance without a positive SFT signal.

## Held-Out Token-Matched Sample-More Baseline

Token-matched sample-more on the same held-out direct pool completed, using the frozen repair budget as the target:

- Target forward-token budget: 79,614.
- Actual estimated forward tokens: 79,861.
- Extra direct-sampling calls: 344.
- Hidden coverage: 65.3% (98 / 150 tasks).
- Zero-to-one additions: 5 / 57.
- Zero-to-one rate: 8.8%.
- Mean candidates/task: 5.18.
- Mean hidden-pass candidates/task: 2.39.
- Mean distinct behavior rate: 0.708.

Interpretation note: at matched estimated model-forward-token cost, spending the budget on additional direct samples beat both repair arms on the headline metric: sample-more recovered 5 zero-base tasks, frozen repair recovered 3, and SFT repair recovered 2.

## Analysis And Report

Commit-policy summaries were generated for direct, frozen repair, SFT repair, and token-matched sample-more pools under first-visible, public-signature-majority, shortest-visible, and oracle-coverage policies.

The final report was generated at `reports/qwen35_4b_trained_vs_frozen_repair_mdp_report.md` with figures under `reports/figures/`.

Final readout:

- Trained repair did not beat frozen repair: SFT recovered 2 / 57 zero-base tasks versus frozen repair's 3 / 57.
- SFT repair had a worse false-repair profile: 29.2% visible-pass-hidden-fail repairs versus frozen repair's 25.0%.
- DPO was skipped by gate.
- Token-matched sample-more was the strongest held-out arm: 5 / 57 zero-base tasks at approximately the same estimated forward-token budget as frozen repair.

Conclusion: this package does not support trained repair as a deployable posttraining lever under the tested small verified-repair recipe. The stronger observed use of extra model budget was additional diverse direct sampling.

## Final Audit

- `python -m py_compile src/*.py scripts/*.py` passed.
- No files larger than 10 MB are present inside the experiment directory.
- Large LoRA artifacts are stored separately under `/workspace/large_artifacts/qwen35_4b_trained_vs_frozen_repair_mdp`.
- Report figures were generated as valid PNG files.
- Generated `__pycache__` directories were removed.
- Standalone-reference scan found no external-paper/citation hooks or cross-track narrative references; only the date-like seed value `20260625` matched the broad scan.
