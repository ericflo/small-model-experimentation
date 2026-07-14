# Search-Scaffold Universal Curriculum Report

## Summary

The design is frozen after successful CPU feasibility. The same-parent exact-token
replay control and staged-search candidate both trained successfully. The scaffold
then failed the single fresh local mechanism gate; no merge or benchmark event ran.

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
because the targets differ.

Fresh paired local seed 88,007 gave parent 18/26 correct, 23/26 parsed, and three
cap contacts; replay 16/26, 23/26, and three; scaffold 16/26, 23/26, and three.
The candidate was 0/2 on execute, 0/2 on induction, and 0/2 on probe. It failed five
of six registered checks—accuracy, parse, cap, execute, and induction—and passed
only route abstention. Promotion is empty, so no checkpoint was merged and aggregate
seed 78,137 remains sealed.

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

The intervention is executable but fails its intended mechanism. Against parent,
candidate has two paired wins and four losses; against replay, three wins and three
losses. Mean generation length is 520.5 tokens versus 434.2 parent and 471.6 replay.
Both candidate execute failures compute the correct final state in visible thought
but continue to the cap without an answer. Both probe cases regress from correct in
both controls to wrong in candidate, and both induction cases are wrong. The package
therefore neither commits after verified execution nor improves branch simulation.

The post-result diagnosis is an interface mismatch: training uses exactly two
canonical-coded operations and a two-branch demonstration, while the local executor
uses natural-language procedures of variable depth and probe selection requires
independent simulation/scoring. This does not justify tuning the observed arm. It
justifies a new result-separated natural-language state-table/compiler mechanism.

## Next Experiments

Preserve and publish this negative. Start a new experiment with fresh seeds if the
next natural-language state-table/compiler mechanism survives idea intake and design
review. Do not lower the gate, reuse seed 88,007, merge this adapter, or consume
aggregate seed 78,137.

## Artifact Manifest

The parent, frozen data identities, commands, and future external-artifact locations
are recorded in `artifact_manifest.yaml`. Both adapters exist externally; the full
local receipt and empty promotion receipt are committed. No merge or benchmark
artifact exists.
