# Backlog

## Next Experiments

- **In progress:** `qwen35_4b_native_thought_seam_budget_ladder` freezes the
  required selection-only natural-close ladder (256/512/1024), smallest-cap
  rule, and fresh same-cap confirmation. Its 32-threat adversarial review
  forbids forced close, a larger-cap confirmation rescue, and pseudo-replicating
  the paired rungs. Any future patch must replay the exact live prefix and
  construct exact random controls dynamically at every sequence length.
- Only after that seam passes, retry the frozen thought-prefix value design:
  continuation labels, held-out-by-task J value, then scalar causal patching
  against exact random, shuffled-axis, identity, logit, ActAdd, raw, and non-J
  controls.
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
