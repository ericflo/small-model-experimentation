# Qwen3.5-4B Context-Local Jacobian Clamp Experiment Log

## Scaffold

Created as a new experiment scaffold.

## 2026-07-12 — intake and adversarial design

- Named `qwen35_4b_jacobian_value_transport` as the closest near-duplicate.
- Registered the material correction: selected-token position, direct-concept
  pullbacks, fixed donor coordinate clamps, and exact realized-norm controls.
- Completed the adversarial design review before implementation or scientific
  GPU inference.
- Prohibited target-digit gradients from intervention construction and band
  selection; they are diagnostic-only after confirmation is frozen.
- Registered a full-activation donor site gate before any J conclusion.

## 2026-07-12 — immutable design boundary

- Pushed design commit `c1f06c035404bde62303439daa66dba3c1f026f9` to
  `origin/main` before any result-bearing model call.
- Recorded exact SHA-256 values for the frozen README and preregistration in the
  config and `runs/design_boundary_receipt.json`.

## 2026-07-12 — cache-free model plumbing

- Implemented selected-token discovery, context-local direct-logit pullback
  fitting, fixed full-activation donor patching, fixed coordinate clamping, and
  additive control patching under batch-one `use_cache=False` forwards.
- Added full-rank SVD diagnostics, exact coordinate/idempotence tests, and
  row-wise span-orthogonal norm-control tests.
- Moved `Key:` / `Value: ` into the assistant response prefix so direct concepts
  and bare digits obey the preregistered one-token contracts.
- CPU suite passes 22 tests plus 24 subtests. No model result has been observed.
