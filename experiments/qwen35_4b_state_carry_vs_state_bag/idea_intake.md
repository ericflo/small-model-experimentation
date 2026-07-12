# Idea Intake: State-Carry Versus State-Bag

## Program Fit

- Program: `structured_execution_and_compilers`
- Existing or new program: existing; the question is explicitly about a latent/stateful runtime and representation.
- Closest program scorecard reviewed: Structured Execution And Compilers.
- Related future queue item: `typed_vs_latent_vs_text_compiler_suite`; this experiment is narrower because it isolates serial state dependence before comparing representation families.

## Prior Evidence

- Anchor 1: C44/C45 — general induction becomes available through load-bearing serial token computation but remains absent in one forward pass.
- Anchor 2: `latent_executor` and `sampled_query_filter_executor` — recurrent computation develops a depth threshold when its state is jointly sufficient; sampled queries can identify such a state without emitting the full trace.
- Anchor 3: C19/C20/C30 and the context-local Jacobian transport line — information can be decoded without being behaviorally usable, while a correctly aligned interface can make semantic content causally consumable.
- Anchor 4: C54 — static weight installation reaches a quick-versus-deep Pareto frontier, motivating runtime depth that is disabled on easy items.
- Closest duplicate or near-duplicate: `qwen_fastweight_hook`. Its 256-dimensional bolt-on, answer-letter supervision, 300 training steps, and noisy 100-item K curves did not produce robust recurrence; larger retests erased the apparent gains.

## Novelty Claim

No prior repository experiment or located paper directly trains and compares a serially carried Qwen state against an equal-parameter, equal-decoder-layer-token, separately optimized ensemble of reset shallow states while also requiring query-independent state sufficiency, unseen-recurrence extrapolation, and donor-consistent causal swaps.

## Mechanism

Repeated computation is useful only if later computation consumes what earlier computation discovered. A causal workspace before the unknown query, a state-only cross-loop bottleneck, and shared multi-query supervision should turn the repeated block into a stationary state transition. The explanation is false if a separately trained reset-state bag matches Carry, if extra K helps only inside the trained horizon, or if swapping the purported state does not transport the donor's consequence.

## Control Plan

- Baseline: same model, prompt, recurrent block, trainable parameterization, optimizer, data, K, and readout with the state edge reset on every extra call.
- Mechanism-falsifying control: separately trained State-Bag plus an inference-time edge cut of the exact same trained Carry checkpoint, each required to show a positive complete-cell effect rather than mere artifact availability.
- Shift or robustness check: depth 5–12, held-out transition family, held-out surface template, and joint holdout.
- Hidden-label boundary: procedural trajectories may supervise training-state heads, but a disjoint pilot seed/split only decides stop/proceed; confirmation labels never select checkpoints, K, layers, prompts, sampling policy, or thresholds. Interface variants require successor experiments.

## Evidence Output

- Program evidence update: only after a result-bearing run changes belief.
- Claim ledger or synthesis update: only a multiseed terminal verdict may create or amend a claim.
- Reusable artifact: Qwen3.5 recurrent-middle-block wrapper, query-after-state procedural generator, paired swap harness, and equal-compute receipts.
- Stop condition: see `reports/preregistration.md`; early raw gains never skip the trained Bag, crossed-task confirmation, causal edge-cut/swap, joint-holdout, or valid sample-more gates.

## Decision

- Run experiment: yes, after live model smoke.
- Create program: no.
- Write synthesis only: no; the serial-versus-reset uncertainty is genuinely open.
- Defer: expensive stages are phase-gated, not deferred.
