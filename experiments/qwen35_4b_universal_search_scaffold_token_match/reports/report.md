# Search-Scaffold Universal Curriculum Report

## Summary

The design is frozen after successful CPU feasibility. The same-parent exact-token
replay control and staged-search candidate have both trained successfully; all
capability evaluation remains pending. No capability result exists.

## Research Program Fit

This is the result-separated successor to the close-weight negative. It preserves
the strong emission near-miss but changes mechanism: independently supervised search
substates rather than more close loss or another representative dose.

## Method

The five stages are apply-first, fit-second, reject-first, execute-pair, and bounded
two-branch search. Every target is recomputed by an experiment-local executable
specification over abstract surfaces. The candidate contains 200 inherited replay
rows, 80 scaffold rows (16/stage), and 40 replay fillers. The control contains the
same 200 rows plus 120 replay rows. Both streams have 320 rows, 286,814 forward tokens,
zero skips, and 40 frozen updates from the same authenticated parent.

## Results

CPU construction passed: source SHA-256 `5854c218...a093`; candidate stream
`79a8d7c9...0b90`; replay stream `c157fb13...355d`; exact token receipt
`eeb12b95...e4a0f`. Forty-three experiment tests and the full smoke harness pass.
The replay control completed 40/40 updates over 320/320 rows with zero skips in
281.2 seconds. Final train loss was 0.4215. Its adapter weights/config SHA-256 are
`10155232...fc538` / `373c1426...ac9b`; receipt/log SHA-256 are
`5b293eb6...5a66` / `7d3bc262...d5f7`.

The scaffold candidate then completed 40/40 updates over 320/320 rows with zero
skips in 291.4 seconds. Final train loss was 1.492. Its adapter weights/config
SHA-256 are `e7957d90...84618` / `22859c76...2c4ce`; receipt/log SHA-256 are
`13ba8897...6dd0` / `ccaffa7b...99c1`. The losses are not a capability comparison
because the targets differ. No local evaluation, merge, or benchmark event has run.

## Controls

- Authenticated `close_xi` parent.
- Authenticated newly trained replay-only continuation from that parent.
- Exact row, forward-token, update, seed, optimizer, close-weight, and parent matching.
- Two hundred byte-identical replay positions; batch size one avoids padding-compute
  differences.
- Fresh local procedural seed 88,007 before conditional aggregate seed 78,137.

## Oracle Versus Deployable Evidence

Executable generators may use hidden construction state only to truth-audit rows.
Promotion uses autonomous greedy model behavior. Benchmark access remains behind the
aggregate-only firewall.

## Interpretation

The intervention is now executable and causally sharper than another generic dose:
only the 120 variable slots differ. Exact forward compute does not equalize target
composition—the candidate has more prompt/answer and fewer thought tokens—and the
full-search targets jump from one known-dead branch to the true branch. Those are
registered interpretation limits, not hidden after-the-fact caveats.

## Next Experiments

Publish and verify the completed candidate, then run the single fresh local event
over parent, replay control, and scaffold candidate. Consume the paired aggregate
pilot only if the sole candidate passes every frozen local check.

## Artifact Manifest

The parent, frozen data identities, commands, and future external-artifact locations
are recorded in `artifact_manifest.yaml`. Both adapters exist externally; no local,
merge, or benchmark artifact exists yet.
