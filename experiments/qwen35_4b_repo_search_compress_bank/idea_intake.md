# Idea Intake

## Program Fit

- Program: `agentic_breadth_installation`
- Existing or new program: existing
- Closest program scorecard reviewed: `knowledge/program_scorecards.md`
- Related future queue item: C53's queued scaffold-distillation of tool-found solutions

## Prior Evidence

- Anchor 1: C12/C22 — tool search followed by verified banking can extend the fixed model's compositional frontier.
- Anchor 2: C54 — compression advantage can install deeper procedures when ordinary self-distillation has saturated.
- Anchor 3: C52 round 2 — the closest repository tool loop is evaluation-only; it contains no trajectory bank or coding curriculum.
- Closest near-duplicate: `qwen35_4b_think_ftpo_round2`, which evaluates six procedural mini-repository families but trains only token pivots on a different substrate.
- Main negative constraint: `qwen35_4b_interactive_policy_curriculum`, where full-sequence DAgger collapsed VERIFY/COMMIT because only 55 of 2,270 targets were VERIFY.

## Novelty Claim

This is the first training-time repository search→verify→compress→bank loop in the corpus. It samples multiple real tool trajectories, accepts only private-test successes, deletes replay-unnecessary patches, reconstructs compact executable traces, and equalizes INSPECT/PATCH/VERIFY/COMMIT loss mass. It is not full-trajectory imitation, token-pivot steering, or another same-recipe breadth round.

## Related Claims

- C12/C22: verified tool-seeded banking can cross depth walls.
- C50/C53: broad emission-policy installation transfers, but the recipe has saturated.
- C54: compression advantage is the strongest current serial-compute mechanism.
- C52: entropy/varentropy can route acquisition and diagnose forks, but must not determine token pressure.

## Mechanism

Successful long agent trajectories contain useful search discoveries but also redundant, failed, and operator-skewed actions. Executable replay compression should retain the causal repair and its verification/commit protocol while removing imitation noise; balancing the four operator classes should prevent patch-heavy semantic capture. The mechanism is false if action-only training performs equally, replay compression does not transfer to wholly new repository families, or the candidate cannot beat a matched-call sampling policy.

## Control Plan

- Baseline: regenerated C54 apex policy, plus the existing C53 blend as a contextual incumbent.
- Mechanism-falsifying control: identical bank and optimizer budget with compact-plan loss set to zero (`action_only`).
- Matched-compute control: two independent four-turn apex trajectories versus one eight-turn trained trajectory, with the same maximum model calls and reserved sampled tokens.
- Shift check: six generated training families and four family-disjoint transfer families, followed by a second fresh transfer block.
- Hidden-label boundary: hidden executable source and oracle edits remain host-only; serialized trajectories receive only terminal booleans and digests.

## Evidence Output

- Program evidence update: result and whether compact, operator-balanced banking earns another iteration.
- Claim ledger or synthesis update: only after a gated, replicated result; no claim id is reserved while the concurrent Pareto-integration run owns its question.
- Reusable artifact: procedural repository generator, constrained coding harness, replay minimizer, operator-balanced bank builder, and receipts.
- Stop or branch condition: no Menagerie exposure unless transfer, matched-sampling, operator-retention, and locality gates all pass. A failed compact-vs-action-only comparison ends this recipe rather than inviting dose tuning on evaluation outcomes.

## Decision

- Run experiment: yes, staged and preregistered.
- Create program: no.
- Write synthesis only: no.
- Defer: Menagerie remains deferred until every non-benchmark gate passes.
