# Experiment Log: Qwen3.5-4B Diversity-Keyed Coverage Gate

Date: 2026-06-25

Experiment directory:
`/workspace/experiments/qwen35_4b_diversity_keyed_coverage_gate`

Large artifact directory:
`/workspace/large_artifacts/qwen35_4b_diversity_keyed_coverage_gate`

## Objective

Before training any strategy-token or diversity-keyed adapter, measure whether held-out coding misses are diversity-limited or capability-limited.

The headline question is: among held-out MBPP tasks where a small base candidate pool contains no hidden-correct solution, how many become covered when we spend more sampling budget under default, hot, and tuned-diverse decoding?

## Design Commitments

- Use Qwen3.5-4B only.
- Keep this package standalone: its own config, scripts, logs, reports, data, and figures.
- Do not train a LoRA unless the high-K diagnostic shows meaningful recoverable headroom.
- Use hidden tests for evaluation only, never in prompts.
- Measure functional diversity with per-test failure bitstrings, not only surface/AST diversity.
- Compare tuned diverse decoding against default sampling before crediting any future training objective.
- Run frozen repair only as a combined inference-time recipe after sampling, and track false repairs separately.

## Initial Package

- Created standalone experiment directories.
- Added package-local execution/model utilities.
- Added `sample_base_pool.py`, `sample_zero_base_ladder.py`, and `run_frozen_repair.py`.
- Added per-test failure-bit evaluation for MBPP candidates so functional diversity is measurable.

## Smoke

Base smoke on 8 MBPP held-out tasks completed:

- Samples/task: 4.
- Hidden coverage: 75.0%.
- Zero-base tasks: 2.
- Mean candidates/task after dedupe: 3.38.
- Mean functional diversity rate: 0.385.
- Estimated forward tokens: 6,122.

Hot-ladder smoke on the 2 base-missed tasks completed:

- Extra samples per zero-base task: 4.
- Hidden coverage stayed 75.0%.
- Zero-to-one: 0 / 2.
- Estimated extra forward tokens: 2,004.

Smoke decision: scripts, manifests, execution, and functional-diversity fields are working. Proceed to main diagnostic with staged sampling.

## Main Baseline: K=4 Direct Sampling

Completed `main_base_k4` on 80 MBPP held-out tasks.

- Candidate samples/task: 4.
- Hidden coverage: 56 / 80 = 70.0%.
- Base-missed denominator for the gate: 24 / 80 tasks.
- Base-missed task IDs: 15, 16, 22, 26, 31, 34, 35, 36, 39, 42, 43, 44, 48, 55, 59, 60, 67, 70, 73, 77, 81, 83, 84, 87.
- Mean deduped candidates/task: 3.45.
- Mean visible-pass candidates/task: 2.26.
- Mean hidden-pass candidates/task: 1.88.
- Mean behavior diversity rate: 0.756.
- Mean functional diversity rate: 0.477.
- Estimated forward tokens: 69,645.

Decision: the zero-base denominator is large enough for the intended gate. Proceed to K~32 ladder arms on only those 24 missed tasks: default-more, hot, and tuned-diverse decoding.

## Main Ladder: Default-More K~32

Completed `main_default_extra_k32`: added 28 default-style samples to each of the 24 base-missed tasks.

- Hidden coverage: 64 / 80 = 80.0%.
- Zero-to-one recovery: 8 / 24 = 33.3%.
- Recovered task IDs: 22, 35, 36, 42, 67, 70, 81, 87.
- Mean deduped candidates/task: 10.28.
- Mean hidden-pass candidates/task: 2.21.
- Mean behavior diversity rate: 0.670.
- Mean functional diversity rate: 0.358.
- Estimated incremental forward tokens: 165,846.

Decision: extra sampling produced meaningful zero-to-one recovery. Continue the same K~32 diagnostic for hot and tuned-diverse decoding arms before deciding whether any training objective is justified.

## Main Ladder: Hot K~32

Completed `main_hot_extra_k32`: added 28 high-temperature samples to each of the 24 base-missed tasks.

- Hidden coverage: 66 / 80 = 82.5%.
- Zero-to-one recovery: 10 / 24 = 41.7%.
- Recovered task IDs: 15, 22, 35, 36, 42, 55, 59, 67, 70, 81.
- Mean deduped candidates/task: 11.39.
- Mean hidden-pass candidates/task: 2.31.
- Mean behavior diversity rate: 0.685.
- Mean functional diversity rate: 0.359.
- Estimated incremental forward tokens: 173,698.

Observation: hot decoding recovered two more base-missed tasks than default-more, but the recovered set changed rather than strictly containing default-more. It recovered 15, 55, and 59 that default-more missed, while default-more recovered 87 that hot missed.

## Main Ladder: Tuned-Diverse K~32

Completed `main_diverse_extra_k32`: added 28 mixed-temperature/wide-nucleus samples to each of the 24 base-missed tasks.

- Hidden coverage: 65 / 80 = 81.25%.
- Zero-to-one recovery: 9 / 24 = 37.5%.
- Recovered task IDs: 22, 34, 35, 36, 42, 59, 67, 81, 83.
- Mean deduped candidates/task: 11.04.
- Mean hidden-pass candidates/task: 2.23.
- Mean behavior diversity rate: 0.685.
- Mean functional diversity rate: 0.362.
- Estimated incremental forward tokens: 173,623.

Observation: tuned-diverse underperformed hot by one recovered task but recovered tasks 34 and 83 that hot missed.

## Main Ladder: Union K~32

Merged base K=4 plus all three K~32 ladder arms into `main_union_k32`.

- Hidden coverage: 69 / 80 = 86.25%.
- Zero-to-one recovery: 13 / 24 = 54.2%.
- Recovered task IDs: 15, 22, 34, 35, 36, 42, 55, 59, 67, 70, 81, 83, 87.
- Remaining base-missed task IDs: 16, 26, 31, 39, 43, 44, 48, 60, 73, 77, 84.
- Mean deduped candidates/task: 24.09.
- Mean hidden-pass candidates/task: 2.88.
- Mean behavior diversity rate: 0.636.
- Mean functional diversity rate: 0.340.
- Estimated total forward tokens for merged pool: 582,812.

Decision: the base misses are substantially diversity-limited at this budget. Do not train a diversity-keyed adapter in this package; the no-training tuned sampling baselines are already strong and must be the benchmark for any future training. Run a small frozen-repair pass on the remaining union misses to measure complementarity.

## Combined Recipe: Frozen Repair After Union K~32

Completed `main_union_k32_repair`: one frozen repair attempt from each of up to two visible-failing sources on the 11 union-missed tasks.

- Hidden coverage stayed: 69 / 80 = 86.25%.
- Zero-to-one recovery relative to union misses: 0 / 11.
- Visible-passing repairs: 2.
- Hidden-wrong visible-passing repairs: 2 / 2 = 100%.
- Estimated repair forward tokens: 12,774.

Decision: frozen repair is not a useful complement on this slice. It added no hidden-correct candidates and introduced visible-pass/hidden-fail failures. Since the K~32 union still leaves 11 misses, run one adaptive high-budget sampling extension on the remaining misses to approximate a K~128 diagnostic.

## Adaptive High-Budget Extension: Union Hot K~128

Completed `main_union_hot_extra_k128`: added 40 hot samples to each of the 11 tasks still uncovered by the union K~32 pool.

- Hidden coverage: 71 / 80 = 88.75%.
- Zero-to-one recovery relative to union misses: 2 / 11 = 18.2%.
- Cumulative zero-to-one recovery relative to base misses: 15 / 24 = 62.5%.
- Newly recovered task IDs: 73, 84.
- Final remaining base-missed task IDs: 16, 26, 31, 39, 43, 44, 48, 60, 77.
- Mean deduped candidates/task: 29.31.
- Mean hidden-pass candidates/task: 2.91.
- Mean behavior diversity rate: 0.629.
- Mean functional diversity rate: 0.338.
- Estimated incremental forward tokens: 115,971.

Decision: more sampling still recovers some residual misses, but with diminishing returns. Stop generation here and write the standalone report. The main finding is that many K=4 misses are diversity-limited under inference-time sampling, while a smaller residual appears harder at the tested budget.
