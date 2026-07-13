# Idea Intake: Full-Rank Extra-R Delta Capacity Test

## Program fit

- Program: `structured_execution_and_compilers`.
- Existing program: yes. This is a latent recurrent execution question, not a
  new training or scaling program.
- Closest near-duplicate: `qwen35_4b_state_carry_vs_state_bag`, whose valid
  rank-32 LoRA pilot stopped on failure to form the preregistered joint state.
- Earlier negative: `qwen_fastweight_hook`, a much thinner recurrent adapter.

## Why this is a separate experiment

The parent preregistered full-rank extra-R deltas as the mandatory capacity
follow-up for exactly the observed valid deep-state-formation failure. Adding a
new parameterization to the result-bearing parent would mix evidence and violate
the experiment lifecycle. This directory preserves the parent as a terminal
LoRA result and changes one mechanism in a clean successor.

## Hypothesis and falsifier

Hypothesis: LoRA's low-dimensional update subspace prevented the repeated native
Qwen block from learning a jointly sufficient state transition. Direct
full-shape deltas on the same extra R calls will exceed the pilot's joint-state
gate and then allow serial Carry to beat matched-compute Bag.

The hypothesis is false for this design if a mechanically valid full-rank pilot
still fails joint state formation. An OOM, base-path mismatch, wrong target set,
data-parity failure, or incomplete receipt is an implementation/feasibility stop,
not evidence about representation capacity.

## Controls and scope

- Same pinned model, task rows, seeds, training order, losses, K=4 training,
  state slots, step encoding, damping, aggregator, Carry/Bag edge, joint holdout,
  edge cut, and bidirectional swaps as the parent.
- Exactly 62 loop-linear FP32 deltas, zero initialized and active only on extra R
  calls; no PEFT merge or ordinary-path adaptation.
- Parent row identity is frozen by canonical decompressed hashes and checked
  directly against parent artifacts when available.
- G4 is not part of this capacity successor. A positive through causal G3 may
  motivate a later deployment comparison, but cannot imply one.

## Decision

Proceed through the gated run. Begin with trigger validation, then exact data
parity and the real G0 feasibility gate; every later phase remains receipt-gated.
