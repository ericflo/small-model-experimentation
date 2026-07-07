# Is the induction wall a knowledge or a serial-compute limit? Serial-compute (C44)

## Motivation
C43 showed answer-only SFT only *partially* installs scrambled-order shift induction (0.40, plateaus below the
0.72 execute ceiling, catastrophically forgets execution). The deepest open question of the arc: is the induction
wall a *knowledge* limit (the model doesn't know how) or a *serial-compute* limit (it can't fit the computation in
one forward pass)?

## Method (review-hardened)
Train the SAME base on plain-words **chain-of-thought** that demonstrates the induction procedure (find the two
positions → derive the shift k=(j−i) mod 10 → apply (p+k) mod 10 → answer). Full eval **matrix**: every arm ×
{forced-digit (1 forward pass), generation (let it reason)} on held-out shift + affine, plus execute
(forgetting), plus base + strategy-hint (procedure handed over, no SFT), plus a CoT faithfulness audit.

## Results (held-out shift, n=200)
| arm | 1 forward pass | with reasoning (generation) |
|---|---|---|
| base | 0.087 | 0.000 |
| base + strategy hint (no SFT) | — | 0.000 |
| answer-only SFT (C43) | 0.40 | — |
| **reasoning-SFT** | **0.010** | **1.000** |

- execute (forgetting): base 0.72 · **answer-only SFT 0.09** · **reasoning-SFT 0.57**
- out-of-family affine (reasoning-SFT, generation): **0.13**

**The dissociation:** reasoning-SFT induces held-out shifts *perfectly* when it can reason (generation 1.00) but
is at *chance* in a single forward pass (forced-digit 0.01). **The CoT is ~100% load-bearing** — induction lives in
the serial tokens, not the weights. The model literally cannot do the induction computation in one forward pass,
even after training. The emitted CoT genuinely computes the position-arithmetic (verified faithful).

## Conclusion
**The forward-pass induction wall is a serial-compute limit, not a knowledge-storage limit.** This resolves C43:
answer-only SFT tried to cram induction into the forward pass → 0.40 + catastrophic forgetting; reasoning-SFT lets
it unroll serially → 1.00 + preserved execution. Connects C38 (thinking rescues induction to 0.50), C13 (broken
mental *simulation*), C42 (multi-step reasoning): the model's core limit is running multi-step computation in a
forward pass; give it serial tokens and it works.

## Honest caveats
- The CoT **hand-codes the shift algorithm**, so this shows the model can *execute a taught serial
  induction-procedure* perfectly (C39's execute-a-procedure, unrolled) — **not** that it learned *general*
  induction: out-of-family affine stays near chance (0.13, shift-specific).
- base + strategy-hint = 0.00 because the position-arithmetic is itself too hard for the *untrained* base to
  execute (a confound), so it doesn't cleanly test serial-compute-without-teaching.
- Single seed; general induction (multi-family leave-one-family-out) is the owed next step.

## Artifact Manifest
See `reports/artifact_manifest.yaml` (adapters external).
