# Backlog

## Next Experiments

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
