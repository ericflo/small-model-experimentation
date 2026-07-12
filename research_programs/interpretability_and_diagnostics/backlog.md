# Backlog

## Next Experiments

- `qwen35_4b_native_thought_seam_budget_ladder` is terminal
  `NO_BUDGET_SELECTED`: all 48/48 traces contacted even the 1,024 ceiling, with
  zero natural closes at every rung. Confirmation was correctly unopened. Do
  not add a larger natural-close rung or treat these rows as completed thoughts.
- **In progress:** `qwen35_4b_forced_commit_jacobian_value_transport` treats
  injected close as the explicit deployed policy. Its 46-threat review first
  gates forced-only parsing, success headroom, mixed tasks, and answer
  termination on selection/confirmation; only then may it label exact prefixes
  and test scalar J value. Every causal replay uses the live prefix-plus-close
  sequence and per-row exact post-bf16 controls. No stage is called natural.
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
