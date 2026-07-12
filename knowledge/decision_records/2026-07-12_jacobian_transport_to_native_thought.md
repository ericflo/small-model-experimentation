# Decision: Move Replicated J Transport Into Native Thought

- Date: 2026-07-12
- Programs: `interpretability_and_diagnostics`, `test_time_reasoning_budget`,
  `structured_execution_and_compilers`
- Experiment: `qwen35_4b_jacobian_transport_control_replication`
- Status: accepted; native-thought successor required

## Context

The late answer-position Jacobian experiment established local writability but
zero downstream transport. Its early context-local successor showed perfect
semantic transport but was formally invalid because one random norm row and
post-bf16 span leakage failed the frozen control standard.

The fresh replication froze that lens and band, developed outcome-blind exact
bf16 controls, passed 480/480 calibration rows, and then reproduced 48/48 direct
and 48/48 separately mapped consequences. Both random arms and the logit-lens
control were 0/48; wrong-donor J produced its own result 48/48; all 960
confirmation control-layer rows passed.

## Decision

Retire “does an early J coordinate transport at all?” as the immediate question.
Move to native `<think>` prefixes on fresh exact-verifier tasks. First establish
whether prefix value/certainty is causally transportable under the same exact
control standard. Then learn a non-oracle controller from training tasks and
require a held-out win over frozen Qwen3.5-4B and matched-compute sampling.

## Alternatives rejected

- More lookup replications: the mechanism is now independently reproduced and
  another identical substrate would not address deployment.
- Directly claim capability: target donor identity is supplied, so the current
  intervention is an oracle state edit.
- Jump straight to training: without a native-thought positive control, a null
  would confound controller learning with absence of a causal thought-state
  mechanism.

## Reversal conditions

Return to representation construction if native high-value donors fail to move
verifier-scored continuations under valid controls. Retire the deployment line
if a learned controller cannot replicate a contamination-free held-out gain over
matched sampling, even when the oracle native-thought mechanism is positive.

## First successor outcome

`qwen35_4b_native_thought_jacobian_value_transport` stopped at
`NO_NATURAL_SEAM`: 48/48 traces hit its preregistered 160-token cap without
natural close, and variable suffix length changed historical-token activations
by up to 0.0625. No prefix value or causal outcome was opened. This does not
reverse the decision to enter native thought; it inserts a required interface
stage. The next successor must select and confirm a naturally closing budget on
fresh data and use per-length dynamic control geometry before revisiting value.

## Second successor outcome

`qwen35_4b_native_thought_seam_budget_ladder` exhausted the separate frozen
256/512/1024 selector with `NO_BUDGET_SELECTED`: all 48/48 fresh traces reached
1,024 without natural close, yielding zero parseable/usable rows at every paired
rung. All cache audits passed, no exact short-period tail loops were detected,
and confirmation correctly remained unopened. The natural-cap repair branch is
closed for this workload.

This does not reverse the decision to test native thought or weaken the
replicated J mechanism. It changes the interface: the next experiment may use
forced close only as an explicit deployed commit action, with fresh
parse/headroom calibration and C51's counterfactual-state label. If that gate
passes, continuation value and causal edits must be evaluated under the same
forced policy with exact-prefix replay and dynamic per-length post-bf16 controls.
