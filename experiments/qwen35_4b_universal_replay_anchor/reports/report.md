# Replay-Anchored Universal Curriculum Continuation Report

## Summary

Complete negative for the designed curriculum. Replay anchoring preserved a clean
local install, but the candidate lost to both the strong starting policy and the
replay-only mechanism control on the paired aggregate event. The replay-only control
became a stronger next anchor with no family regression and eight strict family gains.

## Research program fit

This is the result-separated integration follow-up to
`qwen35_4b_universal_curriculum`. The parent established that 800 truth-audited
designed lessons are installable but, when continued sequentially from `blend` at
`5e-5`, specialize and displace broad behavior. This experiment tests replay anchoring
at a fivefold lower learning rate. Its from-base replay-union control also failed local
parse and cap gates, leaving mature-policy retention as the unresolved geometry.

## Method

Both arms warm-start the immutable C53 `blend` adapter and use the pinned
`Qwen/Qwen3.5-4B`. Each receives 1,520 rows, one epoch, and 190 effective-batch-8
optimizer steps. The candidate substitutes 400 designed rows for 400 replay rows while
sharing the other 1,120 replay rows byte-for-byte with the mechanism control.

Exact source and derived hashes are in `data/dose_manifest.json`. At max length 4,096,
all 3,040 arm-rows encode without a skip. Candidate exposure is 1,231,404 forward
tokens; replay-control exposure is 1,444,589.

## Results

`warm_union` completed 1,520/1,520 rows, zero skips, 190 steps, finite loss 0.7727, and
an authenticated nonzero adapter. On seed 88,003 it achieved 0.7308 exact accuracy,
0.9615 parse rate, 1/26 cap contacts, and zero feasible-route abstentions, passing all
four local gates. Per-kind residuals include induction 0/2 and state 0/2; local success
does not establish universality.

`replay_refresh` completed 1,520/1,520 rows, zero skips, 190 steps, and finite loss
0.4365. Its adapter SHA-256 is `c296c774...d36a`, versus `26837fad...8f18` for the
candidate. Both were explicitly merged. The candidate merged weight SHA-256 is
`29baf3ad...22f6`; the replay-control merged weight SHA-256 is `22c61ceb...bc9e`.
On frozen quick@1,024 seed 78,133, aggregate scores were base 0.1750, `blend` 0.4410,
`replay_refresh` 0.4851, and `warm_union` 0.4238. The candidate was +0.2488 versus
base, -0.0172 versus `blend`, and -0.0613 versus replay refresh. Its `rites` score
fell 0.125 below base; only five families strictly improved. It therefore failed the
no-negative-family, every-family-positive, strong-control, mechanism-control, and
overall pilot gates.

`replay_refresh` beat base by 0.3101 and `blend` by 0.0441. All ten family deltas
versus base were nonnegative and eight were strictly positive; `rites` and `sirens`
tied base. This is not the registered all-family outcome, but it is a material control
result and the correct anchor for a result-separated successor.

## Controls

- Immutable base and C53 `blend` strong controls.
- Optimizer-step-matched replay-only mechanism control with more token compute.
- Fresh local seed 88,003 and, conditionally, aggregate-only quick@1,024 seed 78,133.
- Explicit adapter merges and one `qwen_vllm` backend for every benchmark arm.
- No benchmark item, transcript, verifier detail, result detail, or raw stream crosses
  the trusted gateway.

## Interpretation

The designed signal remained locally learnable at low rate, but the arm containing 400
designed rows transferred worse than replay alone. The candidate does not establish a
universal feature. Because replay refresh had 17.3% more forward-token exposure despite
matched optimizer steps, this comparison rejects the candidate but does not isolate
designed content as the cause of the full gap. The replay control still shows that the
mature policy was improvable by continued diverse practice; broad replay is an active
capability baseline, not a neutral retention ingredient.

## Next experiments

Use `replay_refresh` as the immutable anchor in a new experiment. Target the two tied
families through abstract, contamination-free procedures only after a fresh local
qualification, while requiring retention of all eight strict gains. Match both optimizer
steps and forward-token exposure against replay continuation, and retain matched-compute
sampling in confirmation on a fresh aggregate seed. Do not retune this result-bearing
directory or reuse seed 78,133.

## Artifact manifest

See `artifact_manifest.yaml`; planned entries are replaced with authenticated checksums
immediately after training and merge.
