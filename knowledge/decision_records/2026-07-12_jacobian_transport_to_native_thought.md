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
