# Backlog

## Next Experiments

- Replicate the invalid-but-perfect context-local J result on fresh mappings at
  fixed band 4–8. Freeze the prior lens; require every post-bf16 random delta to
  match J norm within 1e-5 **and** have realized J-span projection <=1e-3, with
  multiple independent random draws. Native-thinking work remains forbidden
  until this fresh control replication passes.
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

Do not promote the 48/48 context-local transport result while its frozen verdict
is `INVALID_CONTROL`; repair the measurement on fresh items, not in place.
