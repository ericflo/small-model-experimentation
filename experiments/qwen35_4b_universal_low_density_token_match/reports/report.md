# Low-Density Token-Matched Universal Curriculum Report

## Summary

Completed negative. All three exact-token arms trained cleanly, but none passed the
fresh local gate. The benchmark remained sealed.

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

All arms completed 190/190 updates over 1,520 rows with zero skips. Final training
losses were 0.4069 (`replay_repeat`), 0.5128 (`designed40`), and 0.5864
(`designed80`); wall times were 1,380.519, 1,362.717, and 1,334.550 seconds.

On fresh local seed 88,004, replay repeat and the 40-row arm each scored 0.500
accuracy and 0.538 parse with 13 and 12 cap contacts. The 80-row arm scored 0.538
accuracy, 0.615 parse, and 10 cap contacts. The inherited replay-refresh anchor was
0.538 / 0.577 / 11. Every candidate missed the registered accuracy, parse, and cap
bars, while all passed the feasible-route abstention check. No arm was eligible;
explicit merge and paired benchmark stages did not run.

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

The exact-token comparison shows that 40 and 80 designed rows are below the local
installation threshold from this anchor. The 80-row arm directionally improved
parseability and cap behavior over replay repeat, but it did not improve accuracy
over the inherited anchor and remained far below every absolute gate. This is a
local mechanism negative, not a broad-retention measurement.

## Next Experiments

Use a new result-separated experiment. The next design should explicitly bridge the
large gap between the locally passing 400-row parent and the locally failing 80-row
arm, or target concise answer commitment directly while preserving an exact-token
replay control. It must use fresh local and benchmark seeds and keep the benchmark
sealed until a prospectively frozen local mechanism gate passes.

## Artifact Manifest

See `artifact_manifest.yaml`. All adapters and local receipts are authenticated;
no merged checkpoint was produced.
