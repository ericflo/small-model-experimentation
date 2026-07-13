# Post-Model Smoke 002 Audit

Completed after outcome-blind `MODEL_SMOKE_PASS`. No logits, probabilities,
correct aliases, hidden examples, target pipelines, or outcomes were retained.

## Receipt

- Cross-probe suffix lengths are equal and the anchor activation difference is
  exactly `0.0`, clearing the unchanged `0.001` gate.
- All 20/20 live non-J rows pass.
- Maximum realized norm error is `9.3031e-6` <= `1e-5`.
- Maximum realized full-J-span projection is `0.0099543` <= `0.01`.
- Maximum lattice repair steps is zero.
- All 60 outcome-blind full/J/mean/additive/wrong/logit intervention rows pass.
- Source/donor positions and lengths, piecewise/whole tokenization, exact-once
  hooks, donor immutability, lens/model/config hashes, and boundaries pass.
- Peak allocated GPU memory is 8.61 GB.
- The exact 512-token prefix is unchanged from failed smoke 001, SHA-256
  `92d2453ef64981746f708e238f4f4560ebf41e4b8c4aa1369d6e3364a9f6fc81`.

## Adversarial interpretation

The repair changed only post-anchor suffix shape. Identical prefix IDs and
numeric geometry before/after rule out a seed or intervention rescue. Moving
from `0.078125` to exact zero at equal length directly confirms the hybrid
full-sequence shape artifact and preserves, rather than relaxes, causal
invariance.

This is plumbing evidence only. It says the representative row is numerically
feasible and every patcher runs at the right token. It contains no indication
that J, text, full donor, or any other arm selects a target or consequence.

## Authorization

After this complete receipt is committed and pushed, authorize the single full
outcome-blind calibration over four tasks, 11 donors, two probes, two controls,
and five layers: exactly 880 numeric rows. It must reuse the exact task-zero
prefix IDs above, preserve all four prefix ID lists, exercise every intervention
arm, and retain no logits/probabilities/outcomes. Mechanics remains forbidden
until the full receipt is separately audited, committed, pushed, and hash-locked.
