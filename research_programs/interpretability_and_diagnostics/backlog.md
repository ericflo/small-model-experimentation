# Backlog

## Next Experiments

- `qwen35_4b_native_thought_seam_budget_ladder` is terminal
  `NO_BUDGET_SELECTED`: all 48/48 traces contacted even the 1,024 ceiling, with
  zero natural closes at every rung. Confirmation was correctly unopened. Do
  not add a larger natural-close rung or treat these rows as completed thoughts.
- `qwen35_4b_forced_commit_jacobian_value_transport` is terminal
  `FORCED_COMMIT_SEAM_FAIL`: forced-only parse was 12.5%--18.8%, exact success
  1/48 at every cap, and 85%--96% of answers exhausted 16 tokens, usually by
  restarting analysis. An EOS-tolerant parser diagnostic stayed <=22.9%. No cap,
  confirmation, value fit, or J causal outcome opened.
- `qwen35_4b_commit_slot_jacobian_value_transport` is terminal
  `COMMIT_SLOT_SEAM_FAIL`: fixed syntax repaired answer mode and the 1,024 arm
  beat no-thought/shuffled by +6.25pp/+8.33pp, but only five tasks mixed outcomes
  versus six required and task-bootstrap intervals crossed zero. J value stayed
  sealed.
- **In progress:** `qwen35_4b_commit_slot_semantic_power_replication` freezes
  cap 1,024 and uses the calculated 113 fresh tasks per seam stage (~80% power
  at the parent ordered-over-shuffled effect), a task-bootstrap lower bound,
  28 mixed tasks, and eight-alias success/choice support. It keeps no-thought as
  a +3pp point gate and opens value only after an identical untouched pass.
  Qualification passed strongly (92/339 real, 46/339 shuffled, task lower
  +8.85pp, 32 mixed tasks, 11/12 correct/chosen breadth); confirmation is next.
- If native thought-state transport passes, train a non-oracle prefix controller
  and require a replicated held-out capability gain over frozen Qwen3.5-4B and
  matched-compute sampling. Oracle donor selection is a mechanism control, not
  the endpoint.
- Build a standard failure-slicing template by operator, family, length, parse status, and evidence state.
- Add attribution and ablation reports for high-performing compiler and selector lines.
- Compare token-pressure and execution-pressure diagnostics across tasks.
- Create small diagnostic probes that can run before expensive training.
- Track when diagnostics change the next experiment, not just describe a result.

## Required Controls

- Ablation tied to a named hypothesis.
- Negative examples and false positives included.
- Diagnostic result connected to a decision.

## Stop Conditions

Do not add diagnostics that cannot change an experiment decision or falsify an explanation.

Do not treat a coordinate that controls the next reported token as a reasoning
variable until a separately computed consequence changes under a matched control.

Do not relabel the replicated 48/48 oracle context-local transport result as a
capability gain. Target identity and clean donor coordinates are supplied; a
native, non-oracle controller must earn deployment separately.
