# Idea Intake: Native-Thought Jacobian Value Transport

## Program Fit

- Program: `interpretability_and_diagnostics`.
- Conditional programs after an oracle causal pass: `test_time_reasoning_budget`
  and `structured_execution_and_compilers`.
- Closest scorecard: move the replicated context-local mechanism into native
  thought before learning a deployable controller.
- Related queue item: `thinking_budget_controller`, but this experiment tests a
  causal state mechanism rather than budget allocation.

## Prior Evidence

1. `qwen35_4b_jacobian_transport_control_replication` independently established
   that the frozen early 24-token J coordinates transport a prompt-local concept
   through a later computation under exact post-bf16 controls.
2. `qwen35_4b_jacobian_value_transport` preregistered thought-prefix G1/G2 but
   correctly cancelled it when its late answer-position lens failed transport.
3. `qwen35_4b_activation_steering` found a decodable first-operation signal but
   inert mean-difference ActAdd; `qwen35_4b_probe_to_prompt` showed that an oracle
   concrete first-operation hint can elicit downstream depth-2 behavior.
4. `qwen35_4b_thinking_separability_probe` decoded correctness only at the answer
   token on contaminated MBPP and found a shuffled-thinking confound. This study
   instead uses fresh procedural tasks, natural in-thought checkpoints, and
   causal intervention.
5. `qwen35_4b_prefix_value_guided_search` found an oracle code-prefix efficiency
   hint but no coverage expansion. It did not edit a prefix state or use J space.

Closest near-duplicate: `qwen35_4b_jacobian_value_transport`. The material delta
is that G0 is now supplied by a separately replicated context-local lens/clamp,
and the primary object is a task-general scalar continuation-value coordinate
inside that frozen J space.

## Novelty Claim

No prior experiment asks whether continuation success is both decodable and
causally writable at a natural token inside Qwen3.5-4B's own `<think>` span using
a context-local J coordinate whose downstream transport has already passed.

## Mechanism

For each natural thought prefix, map the last-prefix-token activation into the
frozen 24-concept J coordinates at layers 4--8. Fit a cross-task scalar value
axis from disjoint continuation success. If this axis is a consumed certainty
state rather than a correlational trace marker, setting a low-value prefix's
scalar coordinate to a same-task high-value donor should improve fresh-seed
continuations while shuffled-axis, exact random, logit-lens, and non-J controls
do not.

The explanation is false if J-space value is not held out by task, shuffled
labels perform similarly, only full answer-identity coordinates work, or the
scalar edit changes verbalization without improving exact first-operation
identification.

## Control Plan

- Baseline: unpatched low-value natural prefixes under identical full-prefix,
  batch-one, cache-free continuation seeds.
- Mechanism-falsifying controls: two exact post-bf16 random arms, shuffled value
  axis, logit-lens value axis, correct-alias identity clamp, wrong-task donor,
  raw donor, sparse J donor component, matched non-J remainder, and ActAdd.
- Shift check: fresh causal tasks and continuation seeds disjoint from value
  fitting and donor selection.
- Hidden-label boundary: ground-truth first operation and selection continuation
  outcomes may label oracle mechanism stages only. No future non-oracle
  controller may receive them at test time.

## Evidence Output

- Program/synthesis update at every terminal gate.
- No claim ID while the repository claim re-grade remains open.
- Reusable natural-prefix full-recompute patch runner and post-bf16 controls.
- Branch: an oracle scalar causal pass creates a separate learned non-oracle
  experiment; any other outcome preserves the failure mode and changes the next
  mechanism rather than tuning this confirmation.

## Decision

Run as a separate result-bearing experiment. The CPU generator already rejected
non-identifiable first-operation targets (notably algebraically reorderable
`negate`-first compositions) before any model call.
