# Idea Intake: Autonomous Close-Weighted Commit Seam

## Intake

- Date: 2026-07-13
- Program: `agentic_breadth_installation`
- Parent result: `qwen35_4b_universal_mid_density_token_match`
- Closest near-duplicate: C51, `qwen35_4b_answer_potential_trace_sft`.
- Other anchors: C50 `qwen35_4b_gauntlet_breadth_round1`,
  `qwen35_4b_tokenizer_eos_answer_commit_factorial`, and the parent exact-token
  0/160/240 ladder.

## Idea

Continue the authenticated designed160 near-miss for only 40 updates while comparing
exact-token replay, ordinary targeted execute/induct SFT, and byte-identical targeted
SFT whose natural `</think>` span is weighted like the answer rather than like the
thought.

## Novelty claim

This is the first experiment in the line to make the model's own natural close event
an independently weighted training target under a byte-identical standard-SFT
control and an exact-forward-token replay continuation.

## Why this is not C51

C51 scored canonical-answer likelihood after injecting a close into overwhelmingly
cap-bound traces. Its forced-close deployment parsed only 13.2%, so the measured
answer state was one the policy rarely reached; its gate correctly cancelled SFT.
This experiment neither scores nor trains an answer only after an injected close.
It trains the autonomous close token at the verified natural end of fresh synthetic
chains, then evaluates whether the policy closes unaided on fresh tasks.

## Why this is not another dose ladder

The parent already located a nonmonotonic optimum near 160 generic designed rows.
This follow-up preserves that installed parent and adds a small failure-targeted
continuation. The standard and close arms use identical new data, so their contrast
tests signal placement rather than data density. Replay repeat falsifies a generic
benefit from 40 more optimizer updates.

## Falsifier and evidence output

The mechanism is falsified at this dose if `close_xi` does not improve the fresh
local closure/parse profile over byte-identical `standard_xi`, or if any apparent
gain is matched by exact-token replay. The evidence output is a complete three-arm
training receipt plus one paired fresh local receipt. Aggregate-only benchmark
evidence is conditional on the unchanged local gate; a pilot still cannot establish
a universal claim without independent confirmation and matched-compute sampling.

## Decision

Proceed with construction seed 77,110, training seed 44, fresh local seed 88,006,
and conditionally sealed aggregate seed 78,136. Commit and push this design freeze to
`main` before any scientific GPU work.
