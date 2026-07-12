# Idea Intake: Forced-Commit Jacobian Value Transport

## Program Fit

- Programs: `interpretability_and_diagnostics` (primary) and
  `test_time_reasoning_budget` (secondary).
- Existing or new program: existing; this connects a replicated causal
  coordinate mechanism to an explicit test-time commit controller.
- Closest scorecard: native-thought value remains untested because autonomous
  termination failed twice.
- Related queue item: `thinking_budget_controller`; this experiment tests the
  state mechanism behind a fixed commit policy, not yet a learned controller.

## Prior Evidence

1. `qwen35_4b_jacobian_transport_control_replication` established exact-control
   48/48 context-local semantic transport in the frozen 24-coordinate J space.
2. `qwen35_4b_native_thought_jacobian_value_transport` stopped at 160 tokens and
   opened no value or causal result.
3. `qwen35_4b_native_thought_seam_budget_ladder` then found 0/48 natural closes
   even at 1,024, with no exact short-period tail loops. The natural-cap branch
   is closed on this workload.
4. `qwen35_4b_answer_potential_trace_sft` (C51) found a modest signal behind an
   injected close but only 13.2% fresh answer parsing, proving that a forced seam
   must pass deployment-shaped parsing before it can be valued.
5. `qwen35_4b_thinking_budget_scaling` used forced close as an actual deployed
   budget policy; an artificial state is legitimate only when the same action is
   part of deployment.

Closest near-duplicate: `qwen35_4b_native_thought_jacobian_value_transport`.
The material delta is that autonomous close is no longer claimed or required:
the injected close is the explicit policy under calibration, labeling,
intervention, and future deployment.

## Novelty Claim

No prior experiment asks whether continuation value under an explicitly
deployed forced-commit policy is held-out-by-task decodable and causally writable
in a context-local J coordinate with exact post-bf16 controls.

## Mechanism

Budgeted thought prefixes often contain partial or complete conclusions while
the model continues rechecking. The fixed close action converts the prefix into
an answer event. If the endpoint carries a task-general “this continuation is
ready/correct” coordinate, disjoint forced-policy rollout success should define
a scalar J axis, and clamping a low prefix toward a high-value score should
improve fresh answers.

The explanation is false if the forced interface does not parse and retain
headroom, J value fails held-out-by-task ranking, shuffled/identity signals
explain it, or exact causal controls perform as well as the scalar edit.

## Control Plan

- Baseline: unpatched low forced-policy prefix with identical replay and seeds.
- Mechanism-falsifying controls: within-task shuffled value axis, two exact
  post-bf16 random arms, correct-alias identity, full-J donor, wrong-task donor,
  logit-lens value, raw donor, J-only donor component, matched non-J remainder,
  and value-fit ActAdd.
- Shift/robustness: disjoint seam selection, seam confirmation, value fit, and
  causal confirmation tasks/seeds.
- Hidden-label boundary: answer correctness fits the value axis and selects
  donors only in oracle stages. No deployable controller receives it.

## Evidence Output

- Update both program ledgers and synthesis at every terminal gate.
- No claim ID while the repository claim re-grade is open.
- Reusable exact-prefix forced replay and per-length control implementation.
- Branch only a replicated scalar causal pass into a non-oracle controller that
  must beat matched sampling on new tasks.

## Decision

Run as a staged separate experiment. CPU task/lens/gate checks pass; the
adversarial review is frozen before any model call.
