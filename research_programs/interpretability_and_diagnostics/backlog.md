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
- **Replicated seam, terminal value negative:** `qwen35_4b_commit_slot_semantic_power_replication` froze
  cap 1,024 and uses the calculated 113 fresh tasks per seam stage (~80% power
  at the parent ordered-over-shuffled effect), a task-bootstrap lower bound,
  28 mixed tasks, and eight-alias success/choice support. It keeps no-thought as
  a +3pp point gate and opened value only after an identical untouched pass.
  Qualification passed (92/339 real versus 46/339 shuffled; lower +8.85pp),
  then confirmation independently passed (98/339 versus 47/339; lower +9.44pp).
  Correct/chosen breadth was 11/12 then 10/12. Resume task-held-out prefix-value
  work only with alias-identity, correct-logit, margin, and dynamic-length
  controls. The one value run then returned `NO_PREFIX_J_VALUE`: shared J AUC
  0.502 (lower 0.442), below slot margin 0.545 and equal-width non-J 0.529.
  Midpoint prospective AUC was 0.608 but endpoint reversed to 0.396, so causal
  work stayed sealed. The permitted phase-specific audit reduced midpoint-only
  J to 0.538 (lower 0.442), below matched non-J 0.600 and tied margin 0.540;
  retire this J-axis successor and redirect to mechanisms that exploit coherent
  thought without assuming scalar J certainty.
- Raw ordered-minus-exact-shuffle probability also failed as a vector-valued
  commit selector: 0.381 beat majority but not confidence/entropy robustly, and
  exact task matching did not beat an oracle-balanced mismatch. Do not tune the
  score; move Jacobian/counterfactual work before the commit to alter proposals
  or continuations under matched compute.
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
