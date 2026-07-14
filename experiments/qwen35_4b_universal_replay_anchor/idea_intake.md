# Idea Intake

## Program Fit

- Program: `agentic_breadth_installation`
- Existing or new program: existing
- Closest program scorecard reviewed: `knowledge/program_scorecards.md`
- Related search: `make related QUERY="warm-start replay-anchored designed synthetic curriculum retain broad transfer"`

## Prior Evidence

- Anchor 1: `qwen35_4b_universal_curriculum`
- Anchor 2: `qwen35_4b_gauntlet_frontier` (C53/C54)
- Anchor 3: `qwen35_4b_interactive_policy_curriculum`
- Closest duplicate: the parent experiment's `blend_then_designed_fast` arm. It
  improved fresh synthetic accuracy and two held-out axes but displaced the incumbent
  broad policy, losing 0.1385 aggregate to `blend`.

## Novelty Claim

This tests integration geometry, not another benchmark-shaped substrate. Every update
rehearses the complete frozen broad corpus while a low learning rate adds the same
truth-audited designed lessons to the mature `blend` adapter. No prior arm combines all
three constraints: mature broad warm start, replay in every continuation epoch, and a
designed executable curriculum disjoint from the benchmark.

## Related Claims

- C14: narrow SFT can capture format and damage unrelated instruction following.
- C53: the broad emission install is a strong one-time step and the correct retention
  control.
- C54: `effskin` warm-start moved target axes but displaced breadth; integration, not
  mere local installability, is the open problem.
- C56: executable exploration transfers, while trace induction can damage easy
  induction; procedure content and integration both matter.

## Mechanism

Trial 1's gradient came entirely from highly templated designed rows. It installed terse
execution behavior locally but overwrote parts of `blend`, the exact C14/C54 failure
signature. Interleaving 74% broad replay while lowering the learning rate should anchor
the mature policy on every optimizer window, allowing only small compatible procedure
deltas to accumulate. The mechanism is false if designed-task gains disappear, if a
replay-only refresh moves identically, or if any held-out family remains below base.

## Control Plan

- Baselines: pinned base, immutable `blend`, and the parent's sequential negative.
- Mechanism control: a replay-only warm continuation matched to the candidate's 190
  optimizer steps, with 400 additional replay rows substituted for designed rows.
- Candidate: warm `blend` plus an exact nested dose of 400 frozen designed rows and
  1,120 shared replay rows, low-rate continuation, exact token receipt, zero skips.
- Hidden-label boundary: only the aggregate gateway; no benchmark source, item,
  transcript, result detail, or raw child stream crosses into the experiment.
- Robustness: fresh synthetic seed before any fresh quick@1,024 benchmark seed; a
  promoted result moves to independent quick and medium@2,048 confirmation.

## Evidence Output

- Preserve every training/local/merge/benchmark receipt and external artifact checksum.
- Update `agentic_breadth_installation/evidence.md` even if replay anchoring fails.
- Do not amend shared synthesis or claims from one quick event.
- Branch again before any score-conditioned curriculum or dose change.

## Decision

- Run experiment: yes, after the parent factorial finishes and the smoke/design gates
  pass.
- Create program: no.
- Defer: train the replay-only control and open benchmark evaluation only after the
  candidate passes its fresh local installability gate.
