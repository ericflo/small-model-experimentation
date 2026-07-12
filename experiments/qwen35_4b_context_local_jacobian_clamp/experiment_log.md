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

## 2026-07-12 — model-smoke batch preflight correction

- The first plumbing-only smoke failed solely because equal-length, unpadded
  batch-two clean logits differed from separate batch-one calls by max 0.21875,
  above the descriptive 0.05 tolerance.
- This is the Qwen hybrid batch-equivalence hazard the design intended to detect.
  The frozen scientific path is already batch-one, so batch equivalence is now a
  recorded diagnostic rather than a blocker. Causal antecedent activations were
  exactly suffix-invariant (max difference 0), all token/position contracts
  passed, all three smoke dictionaries had rank 4, and both patch deltas were
  finite/nonzero.
- No target-answer outcome was inspected or used.

## 2026-07-12 — model smoke passed

- Cache-free batch-one plumbing passed on the pinned Qwen3.5-4B revision.
- All 24 concept tokens and all 10 bare digit tokens are single-token; source,
  target, direct, and consequence selected positions agreed at index 62.
- Causal antecedent activations were exactly suffix-invariant at layers 4, 16,
  and 28. Equal-length batch-two top IDs agreed with batch-one even though full
  logits did not, confirming the registered batch-one policy.
- Small context-local dictionaries were full rank at all three smoke layers;
  coordinate and full-donor patch deltas were finite and nonzero.
- Peak allocated GPU memory was 9.68 GB. This was plumbing-only and did not
  inspect target-answer success.
