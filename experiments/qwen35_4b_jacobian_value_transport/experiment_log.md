# Qwen3.5-4B Jacobian Value Transport Experiment Log

## 2026-07-12 — intake and design freeze preparation

- Synchronized the clean `main` worktree with `origin/main` before creating the
  experiment.
- Ran related-work discovery for Jacobian transport, think-prefix value, causal
  coordinate patching, and counterfactual reflection.
- Selected `interpretability_and_diagnostics` as the primary program and named
  `qwen35_4b_activation_steering` as the closest near-duplicate.
- Separated final correctness from token-level credit: the design uses common-prefix
  sibling continuations and exact rollout value rather than broadcasting a trace
  label across all thought tokens.
- Added a mandatory Qwen-specific positive control before any value or capability
  claim, plus an immutable design boundary before real-model scientific work.
- No result-bearing model call has occurred in this experiment.
