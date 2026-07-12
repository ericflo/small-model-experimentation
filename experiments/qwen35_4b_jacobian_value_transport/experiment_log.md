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

## 2026-07-12 — immutable design boundary

- The design commit was rebased onto concurrent `origin/main` work and finalized
  as `57fe5249e38f1e498e53a63ea4e9a72c0b48e2f0`.
- Frozen preregistration SHA-256:
  `a7e5711236f9c0dd0c39182c1ccad4c881cf18b604247a12f5362faafc627bae`.
- Frozen README SHA-256:
  `119b20dcbb41dbc578cc8aabbbcb7cf65739fc734180654dcfd0f5f69d12fdf0`.
- The run harness now fails closed unless this commit is an ancestor and both
  frozen-file digests match. Scientific GPU work remains unstarted.
