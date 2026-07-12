# Idea Intake

## Program Fit

- Program: `interpretability_and_diagnostics`
- Existing or new program: existing
- Closest program scorecard reviewed: interpretability diagnostics are useful
  only when they change a decision; this experiment has hard causal stop gates.
- Related future queue item: `supervision_causality_ablation` is adjacent but
  addresses supervision placement rather than test-time Jacobian transport.

## Prior Evidence

- Anchor 1: C19 finds first-operation information linearly readable well above
  behavioral access.
- Anchor 2: C20 finds ordinary mean-difference ActAdd inert even with an oracle
  correct-operation direction.
- Anchor 3: C40/C42 find implicit answer and step-local confidence signals, while
  C51 shows an oracle-side trace score can be real yet practically unusable.
- Additional boundary: C52 finds token-local outcome supervision produces
  nonlocal LoRA drift.
- Closest duplicate or near-duplicate: `qwen35_4b_activation_steering`.

## Novelty Claim

Prior work has not tested whether a direction derived from the model's averaged
downstream Jacobian, written by coordinate replacement and validated with local
transport and rollout value, can cross the representation-to-expression gap that
defeated mean-difference activation addition.

## Mechanism

The averaged Jacobian defines token-aligned directions that downstream computation
is generally prepared to read. The context-local Jacobian then measures whether a
specific active coordinate is connected to the future answer in the present trace.
Replacing a competing coordinate with a verifier-aligned coordinate should change
the answer only when both representation and transport are present.

The explanation is false if J-coordinate swaps do not outperform matched random,
wrong-token, logit-lens, raw-donor, non-J/remainder, and ActAdd controls, or if they
only change a directly named answer rather than a downstream consequence.

## Control Plan

- Baseline: frozen Qwen3.5-4B with identical HF backend, full-prefix recomputation,
  token budget, prompt, and decoding seed.
- Mechanism-falsifying controls: logit-lens coordinate swap, mean-difference ActAdd,
  raw donor patch, sparse J component versus non-J remainder, wrong donor,
  norm-matched random, and outcome-shuffled direction construction.
- Shift or robustness check: distinct IID tasks, held string/register families,
  hard depth, multiple seeds, and consequence rather than direct-report effects.
- Hidden-label boundary: exact operations and hidden examples select oracle donors
  and score results only. They are unavailable to any deployable controller.

## Evidence Output

- Program evidence update: whether token-aligned Jacobian coordinates are causal
  on Qwen3.5-4B and whether local transport adds information beyond activation.
- Claim ledger or synthesis update: only after a terminal gate and a fresh pull of
  `origin/main`; preserve a scoped negative if the premise fails.
- Reusable artifact: tested targeted/full Jacobian fitter, coordinate-patching
  hooks, prefix-value sampler, and decomposition/control utilities.
- Stop or branch condition: positive-control failure stops value claims; prefix-
  value failure stops causal patching; a specific causal patch authorizes a new
  non-oracle controller/reflection experiment.

## Decision

- Run experiment: yes, after design review and immutable design commit.
- Create program: no.
- Write synthesis only: no; C20 leaves the stronger causal intervention unresolved.
- Defer: no; the model and GPU path are available and the first gates are bounded.
