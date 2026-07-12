# Idea Intake: Native-Thought Seam Budget Ladder

## Program Fit

- Programs: `interpretability_and_diagnostics` (primary) and
  `test_time_reasoning_budget` (secondary).
- Existing or new program: existing programs; no charter is needed.
- Closest scorecards reviewed: the replicated context-local Jacobian mechanism,
  the failed native-thought seam, and native-thinking budget calibration.
- Related queue item: `thinking_budget_controller`, but this experiment selects
  a measurement interface rather than a deployable budget policy.

## Prior Evidence

1. `qwen35_4b_native_thought_jacobian_value_transport` is the direct parent and
   closest duplicate: all 48 traces hit its frozen 160-token cap, so value and
   causal stages correctly remained unopened.
2. `qwen35_4b_thinking_budget_scaling` found that ordinary native thought often
   needs roughly 512--1024 tokens, but used a public measurement substrate and a
   force-close protocol; it did not establish an autonomous seam for this task.
3. `qwen35_4b_answer_potential_trace_sft` found that 99.37% cap contact at 512
   made teacher-forced answer potential non-actionable, directly motivating a
   natural close/commit gate on fresh data.
4. `qwen35_4b_jacobian_transport_control_replication` established the upstream
   context-local J transport mechanism but did not enter native thought.

Closest near-duplicate: `qwen35_4b_native_thought_jacobian_value_transport`.
The necessary material delta is a separate frozen budget-selection and
untouched-confirmation experiment; changing 160 after observing its failure
inside the parent would invalidate that result boundary.

## Novelty Claim

No prior experiment selects the smallest naturally closing budget on fresh
first-operation-identifiable tasks and then confirms that exact cap on untouched
tasks using the Transformers backend required for the next activation study.

## Mechanism

The 160-token failure should disappear when the model receives its ordinary
native reasoning allowance. If the failure is genuinely an interface budget,
one of the frozen rungs will expose autonomous close-and-commit behavior with
enough correct/incorrect variation for later continuation-value labels.

This explanation is false for the current interface if no rung passes, or if a
selected cap fails unchanged on the untouched confirmation split.

## Control Plan

- Baseline: the inherited 160-token terminal result, used only as lineage—not
  pooled into the new statistics because the cache mode and tasks differ.
- Mechanism-falsifying control: untouched same-cap confirmation; no alternative
  cap may rescue it.
- Shift/robustness: fresh tasks and disjoint seeds; exact prompt and alias grammar
  inherited from the parent.
- Hidden-label boundary: first-operation gold is used only for setup headroom
  and mixed-task gates. There is no training, routing, or capability endpoint.

## Evidence Output

- Update both owning program ledgers and synthesis at the terminal decision.
- Allocate no claim ID.
- Preserve a reusable natural-close selector, cache audit, and frozen cap.
- Branch only `NATURAL_SEAM_REPLICATED` into a new value/Jacobian experiment;
  preserve either negative without relaxing this ladder.

## Decision

Run as a separate experiment. CPU generation and gate reachability pass before
any model call; the adversarial review is frozen with the design.
