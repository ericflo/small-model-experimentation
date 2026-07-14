# Mid-Density Token-Matched Universal Curriculum Report

## Summary

Prepared, not run. This experiment tests a representative nested 0/160/240-row
designed-dose ladder from replay refresh with exact forward-token matching.

## Research Program Fit

This is the direct result-separated follow-up to the 0/40/80 local negative. It
bridges the known gap between an 80-row dose that did not install locally and the
earlier 400-row mixture that installed locally but lost broad capability.

## Method

Every arm shares 1,280 replay rows in the same slots. Three disjoint 80-row designed
blocks each cover all 13 skills and have disjoint replay counterparts with exactly
the same token sums. Replay repeat uses all replay blocks, `designed160` replaces A
and B, and `designed240` replaces A, B, and C. Every arm contains 1,520 rows,
1,405,510 forward tokens, and 190 updates from the same parent.

## Results

Training checkpoint only. All three exact-token arms completed 190/190 updates over
1,520 rows with zero skips. Final training losses were 0.4199, 0.6606, and 0.7284;
wall times were 1,396.686, 1,390.190, and 1,373.185 seconds for replay, 160-row, and
240-row arms respectively. The local screen and paired benchmark remain pending, so
no generalized-transfer comparison is available yet.

## Controls

- Exact-token replay continuation from the same anchor.
- Inherited replay-refresh anchor, `blend`, and pinned base.
- Fresh local seed 88,005 and conditional aggregate-only seed 78,135.
- Explicit merges and one `qwen_vllm` benchmark backend.
- Both doses registered before any model training.

## Oracle Versus Deployable Evidence

Only deployable greedy local outputs and trusted aggregate benchmark fields are
admissible. There is no oracle selector, private-item inspection, or benchmark-shaped
training signal.

## Interpretation

Pending. Local success alone will not be interpreted as generalized transfer.

## Next Experiments

A passing pilot requires independent quick replication, medium@2,048, paired
uncertainty, and matched-compute sampling in a new confirmation experiment.

## Artifact Manifest

See `artifact_manifest.yaml`; planned entries must be replaced with authenticated
checksums immediately after each successful stage.
