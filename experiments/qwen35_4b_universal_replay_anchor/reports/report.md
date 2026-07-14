# Replay-Anchored Universal Curriculum Continuation Report

## Summary

In progress. The replay-anchored candidate completed training and passed every frozen
local gate. The matched replay-only control completed, and both arms have authenticated
explicit merges. The conditional paired aggregate event is next.

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
These completions authorize, but do not prejudge, the frozen paired Menagerie event.

## Controls

- Immutable base and C53 `blend` strong controls.
- Optimizer-step-matched replay-only mechanism control with more token compute.
- Fresh local seed 88,003 and, conditionally, aggregate-only quick@1,024 seed 78,133.
- Explicit adapter merges and one `qwen_vllm` backend for every benchmark arm.
- No benchmark item, transcript, verifier detail, result detail, or raw stream crosses
  the trusted gateway.

## Interpretation

No interpretation before results. A candidate that merely matches `blend`, fails local
designed tasks, loses to replay refresh, or leaves any public family at or below base
does not establish a universal feature.

## Next experiments

A passing pilot must move to a new confirmation experiment with independent quick
seeds, medium@2,048, paired uncertainty, and matched-compute sampling. Any failed pilot
is preserved before changing dose, rate, or curriculum.

## Artifact manifest

See `artifact_manifest.yaml`; planned entries are replaced with authenticated checksums
immediately after training and merge.
