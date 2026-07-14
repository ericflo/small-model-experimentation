# Low-Density Token-Matched Universal Curriculum Report

## Summary

Prepared, not yet run. This experiment tests a nested 0/40/80-row designed-dose
ladder from the stronger replay-refreshed anchor, with exact forward-token matching.

## Research Program Fit

This is the direct result-separated follow-up to the replay-anchor negative. It asks
whether the designed procedures failed because their 26.3% density was too high or
because they provide no broad increment beyond replay continuation.

## Method

Every arm shares 1,440 replay rows in the same training slots. `replay_repeat` adds
two 40-row replay blocks; `designed40` swaps one for a stratified all-skill designed
half; `designed80` swaps both. The replay blocks match the two designed halves at
16,732 and 16,543 forward tokens exactly. Thus all arms contain 1,520 rows and
1,429,053 forward tokens and receive the same 190 updates from the same parent.

## Results

Training checkpoint only. The exact-token `replay_repeat` control completed 190/190
updates over 1,520 rows with zero skips. Final training loss was 0.4069; wall time
was 1,380.519 seconds. `designed40` completed the identical update and token budget
with zero skips; its final training loss was 0.5128 and wall time was 1,362.717
seconds. The 80-row arm, local screen, and paired benchmark remain pending, so no
generalized-transfer comparison is available yet.

## Controls

- Exact-token replay continuation from the same anchor.
- Inherited replay-refresh anchor, C53 `blend`, and pinned base.
- Fresh local seed 88,004 and paired aggregate-only seed 78,134.
- Explicit merges and a single `qwen_vllm` benchmark backend.
- Prospectively registered 40- and 80-row doses with independent local eligibility.

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
