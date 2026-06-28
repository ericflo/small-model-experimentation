# Latent Executor Experiment Log

## Objective

Design and run a stronger experiment along the same line as the Qwen fast-weight adapter work, but with the failure modes addressed directly. The goal is to demonstrate a setting where invisible recurrent latent computation robustly improves accuracy as the recurrent step budget `K` increases, then document the result honestly.

## Lessons From The Prior Qwen Hook Experiment

The previous experiment did not show robust K-scaling. The strongest apparent gains disappeared under larger retesting. Main failure modes:

1. The answer-letter loss was too low-bandwidth.
2. The recurrent loop was a small 256-dimensional bolt-on rather than a necessary execution path.
3. Aggregate 100-example K sweeps were noisy.
4. `K=0` was a trained prompt-conditioned injection, so it was not a frozen-model baseline.
5. Dense intermediate supervision was absent or too weak.
6. The task did not force each recurrent step to correspond to a causal unit of computation.

## New Hypothesis

If the model is trained as a latent neural executor with dense intermediate-state supervision, then accuracy on multi-step modular register programs should improve sharply as `K` approaches the number of program steps. The same recurrent cell should length-generalize beyond the training horizon because it learns a reusable state transition.

Expected evidence:

- For programs of length `L`, final exact-register accuracy should be poor when `K < L`.
- Accuracy should jump when `K >= L`.
- The effect should survive larger paired retests.
- A static one-shot baseline should not show the same length-generalization behavior.

## Experimental Design

Task: two-register modular programs over registers `A` and `B`, modulo `p`.

Candidate operations:

- `A = A + c`
- `A = A - c`
- `B = B + c`
- `B = B - c`
- `A = A + B`
- `B = B + A`
- `A = A - B`
- `B = B - A`

Model:

- Compiler embeds operation type, operand, and position into latent instruction vectors.
- Runtime initializes a latent state from initial `A,B`.
- At recurrent step `t`, the runtime consumes latent instruction `t`, updates hidden state through a GRU-like recurrent cell, activation-gated low-rank operator bank, and optional temporary fast-weight memory.
- Heads predict both register values after each internal step.

Training signal:

- Dense trace loss at every prefix step: predict `(A_t, B_t)` after `t` operations.
- Final evaluation remains exact pair accuracy for `(A_L, B_L)`.

Primary planned comparison:

1. Trace-supervised recurrent latent executor.
2. Static compiler baseline with no recurrent execution.
3. Optional final-only recurrent control if the trace-supervised result works.

Initial train lengths: 1-8 operations.
Hard evaluation lengths: 12, 16, 24 operations.

## Implementation Notes

The first implementation will be a standalone script, not a Qwen hook. This is intentional: the experiment first proves the latent recurrent-computation mechanism under controlled conditions before paying the complexity cost of embedding it back into a frozen LLM.

## Smoke Tests

- `../runs/smoke_executor`: recurrent executor path compiles and runs.
- `../runs/smoke_static`: static baseline path compiles and runs.

Both smoke tests are intentionally too short to learn; they only validate shapes, losses, checkpoint writing, and evaluation output.

## Pilot Plan

Start with `modulus=31`, train lengths 1-8, eval lengths 4/8/12/16/24. This is easier than the previous modulo-97 task but still has a strict exact-pair chance rate of about 0.1%. If the mechanism works, the executor should show high pair accuracy when `K >= L` and low accuracy when `K < L`.

## Pilot 1: Unstructured Hidden-State Executor

Run: `../runs/pilot_executor_mod31`

Stopped manually at step 650 after poor learning. Evidence:

- Trace loss decreased from about 3.61 to about 2.31, so optimization was happening.
- Exact-pair accuracy remained around 1-3% even on length-4 programs.
- No clear K-scaling curve emerged.

Interpretation: the generic hidden-state GRU executor is spending too much capacity discovering a modular arithmetic representation. For a first positive result, we should structure the latent state as categorical distributions over register values and train learned op-conditioned transition operators. That keeps execution neural and latent, but removes a representation-learning bottleneck that is not central to the hypothesis.

## Pilot 2: Categorical Latent Executor, Modulus 31

Run: `../runs/pilot_categorical_mod31`

Configuration:

- `op_family=const`
- train lengths 1-8
- eval lengths 4, 8, 12, 16, 24
- dense trace supervision
- learned transition table over `(op, arg, current_value, next_value)`

Step 200 result:

- `L=4`: accuracy jumps from near chance at `K<4` to 100% at `K>=4`.
- `L=8`: accuracy jumps to 100% at `K>=8`.
- `L=12`: accuracy jumps to 100% at `K>=12`, despite training only on lengths up to 8.
- `L=16`: accuracy jumps to 100% at `K>=16`.
- `L=24`: accuracy jumps to 100% at `K=24`.

Interpretation: this is the first clean positive result. The recurrent latent execution budget is causally necessary: the model is correct exactly when it has enough internal steps to consume the whole latent program. Length generalization works because the learned transition is reused.

Step 400 reproduced the same threshold pattern exactly. The run was intentionally stopped after the saved step-400 checkpoint to move on to controls and harder replication.

## Static Baseline, Modulus 31

Run: `../runs/static_mod31`

Same task distribution as Pilot 2, but with a static Transformer-style compiler and no recurrent execution axis.

Step 200:

- `L=4`: exact pair 0.3%
- `L=8`: exact pair 0.1%
- `L=12`: exact pair 0.2%
- `L=16`: exact pair 0.1%
- `L=24`: exact pair 0.2%

Step 400:

- `L=4`: exact pair 0.2%
- `L=8`: exact pair 0.0%
- `L=12`: exact pair 0.0%
- `L=16`: exact pair 0.0%
- `L=24`: exact pair 0.1%

Interpretation: the one-shot static model does not solve even the short training-like lengths. This supports the claim that the successful categorical executor is using its recurrent transition machinery rather than simply compiling the whole program into a final answer.

## Categorical Latent Executor, Modulus 97

Run: `../runs/categorical_mod97`

Same categorical executor and constant-op task family, but with the original modulo-97 arithmetic scale.

Step 250 result:

- `L=4`: exact pair 100% at `K>=4`; near-zero before enough K.
- `L=8`: exact pair 100% at `K>=8`; near-zero before enough K.
- `L=12`: exact pair 100% at `K>=12`; near-zero before enough K.
- `L=16`: exact pair 100% at `K>=16`; near-zero before enough K.
- `L=24`: exact pair 100% at `K=24`; near-zero before enough K.

Each length used 2,048 held-out examples. Training only used lengths 1-8, so lengths 12/16/24 are genuine recurrent length generalization.

Interpretation: the positive result scales to modulo 97. This is the strongest evidence so far that the revised experiment works.

## Static Baseline, Modulus 97

Run: `../runs/static_mod97`

Step 150 result:

- `L=4`: exact pair 0.0%
- `L=8`: exact pair 0.0%
- `L=12`: exact pair 0.0%
- `L=16`: exact pair 0.1%
- `L=24`: exact pair 0.0%

Interpretation: the matched static baseline is at chance, while the categorical recurrent executor is perfect at `K>=L`. This is a direct control for the modulo-97 setting.
