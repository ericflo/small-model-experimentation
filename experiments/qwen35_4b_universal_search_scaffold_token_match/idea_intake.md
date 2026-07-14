# Idea Intake: Staged Decomposition Search Scaffold

## Intake

- Date: 2026-07-14
- Program: `agentic_breadth_installation`
- Parent result: `qwen35_4b_universal_close_weight_token_match`
- Closest near-duplicate: `qwen35_4b_universal_curriculum`.
- Other anchors: C44, C56, C59, the mid-density exact-token ladder, and the
  close-weight byte-identical negative.

## Idea

Teach two-step rule search as a curriculum of independently scored executable
subproblems before asking the model to compose the full procedure. The five intended
stages are apply a proposed first operation, fit a second operation, reject a first
operation for which no second operation fits, execute a known pair, and solve a full
pair with a bounded compact ledger.

## Novelty claim

This is the first experiment in the universal-curriculum line to supervise the
intermediate search states as tasks in their own right. The original curriculum
taught primitives and then narrated one dead branch plus the correct decomposition
inside a full answer trace. It did not separately train the apply/fit/reject interfaces
that full search must invoke.

## Why this is not another trace or close intervention

C56 and the parent curriculum show that an oracle narration can train to low loss
without deploying a reusable composed-induction circuit. The completed close-weight
experiment further shows that emphasizing `</think>` does not improve parse or cap
behavior over byte-identical ordinary training. This successor changes the supervised
problem distribution: each search transition must itself produce a verified answer.
Close tokens retain ordinary weight.

## Why this is not another dose ladder

The row budget is intended to match the predecessor's short continuation, but the
mechanism variable is lesson decomposition. The same-parent replay continuation
falsifies benefit from another 40 updates. Exact forward-token matching prevents a
shorter structured trace from receiving less compute.

## Falsifier and evidence output

The mechanism fails if the scaffold candidate does not solve at least one fresh
`u_execute` and one fresh `u_induct` case, does not pass the unchanged absolute local
gate, or is matched by replay continuation. The evidence output is a truth-audited
stream receipt, paired training receipts, and one fresh local event. Aggregate-only
benchmark evidence remains conditional and cannot establish universality without
independent confirmation and matched-compute sampling.

## Decision

Proceed only to CPU feasibility and adversarial design review using construction seed
77,111. Reserve training seed 45, local seed 88,007, and conditional aggregate seed
78,137. Commit and push the intake before implementing scientific stages. No GPU or
benchmark work is authorized by this note.
