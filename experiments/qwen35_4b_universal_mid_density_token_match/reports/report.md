# Mid-Density Token-Matched Universal Curriculum Report

## Summary

Completed local negative. A representative nested 0/160/240-row designed-dose
ladder was trained from replay refresh with exact forward-token matching; no arm
passed the fresh local gate, so merge and benchmark remained sealed.

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

All three exact-token arms completed 190/190 updates over 1,520 rows with zero skips.
On fresh local seed 88,005, anchor and replay repeat each scored 17/26 accuracy,
18/26 parse, and 9 cap contacts. `designed160` scored 19/26, 23/26, and 3;
`designed240` scored 17/26, 22/26, and 5. Every arm passed accuracy ≥0.65 and the
route check, but none passed parse ≥0.90 and cap contacts ≤2. Promotion was empty,
so no merge or paired benchmark occurred.

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

The 160-row dose is a real near-threshold local improvement: versus exact-token
replay it adds two correct cases, five parsed answers, removes six cap contacts, and
shortens mean output by about 218 tokens. The effect is not monotonic: 240 rows lose
the accuracy gain and regress parse/cap behavior relative to 160. This rejects
representative dose interpolation as sufficient, not the curriculum signal. The
remaining local bottleneck is concise answer commitment at the 160-row capability
mix. There is no broad-transfer evidence because the benchmark correctly stayed
sealed.

## Next Experiments

Use a new experiment and fresh local seed to hold the 160-row capability mix fixed
while adding a small truth-audited answer-commit/termination intervention with an
exact-token active control. Do not add more representative dose, lower the observed
gate, reuse seed 88,005, or spend aggregate seed 78,135. Any later pilot pass still
requires independent quick replication, medium@2,048, paired uncertainty, and
matched-compute sampling.

## Artifact Manifest

See `artifact_manifest.yaml`; planned entries must be replaced with authenticated
checksums immediately after each successful stage.
