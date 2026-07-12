# Qwen3.5-4B Same-Prefix Advantage Routing Report

## Status

**Terminal negative at the preregistered route-qualification gate.** Both
teachers had adequate support, deep replicated, and the combined router was
positive, but quick's audit advantage over the soup student reversed sign in
block 1. No MOPD or later stage is authorized.

## Research Program Fit

This is the clean state-level successor requested by the agentic breadth and
posttraining programs after `qwen35_4b_pareto_policy_integration` disproved the
assumed quick/deep local route without reaching MOPD.

## Method

The frozen method compares both same-origin teachers and the strongest 40/60
soup student on exact failed student states. Four branches choose a strictly
better teacher or abstain; four disjoint branches estimate selected-teacher
advantage over both student and alternate. Both teachers must independently
replicate before any dense update. Conditional training uses corrected top-k
MOPD, a frozen-soup anchor, exact-logit locality, matched routing/off-policy
controls, two confirmatory blocks, three seeds, visible routing, and best-of-8.

## Evidence

The route study used 384 fresh soup states and 9,216 teacher/student
continuations, plus soup-only acquisition rollouts: 12,726,694 sampled tokens
total. It routed 111 states and abstained on 273. Support passed for quick
(29/26 by block) and deep (22/34).

Deep's audit advantage over the student was `+0.1216` and `+0.0655` by block,
with pooled one-sided 95% LCB `+0.0657`; its alternate-teacher contrast also
passed. Quick beat deep and had pooled student LCB `+0.0677`, but its
student-relative block means were `+0.2009` and `-0.0253`. That negative
replicate is terminal under the frozen rule. The combined router passed, but
composition requires both named teachers to be independently useful.

The key result is methodological: pooling would have declared quick useful and
authorized MOPD, while the independent-block sign gate exposed a nonreplicable
local teacher. The correct response is to preserve the negative, not lower the
bar or train on the favorable block.

## Post-Gate Mechanism Diagnosis

Absolute continuation scores were not generally noisy: selection-to-audit
correlations were `0.79`--`0.86` for all three policies. Conditioning on the
largest of three four-branch estimates was the unstable step. Quick block 1
had an apparent selection advantage of `+0.319` over the student but an audit
state mean of `-0.019`; only 6/26 selected states remained strict quick winners
on audit. Independent halves recovered the same quick route on 12/29 and 6/26
states across blocks.

A fixed positive margin does not solve that winner's curse. Retaining only
quick block-1 states with observed selection margins of at least `0.10` or
`0.25` left audit means of `-0.0259` and `-0.0089`. The six-state `0.50` tail
was positive but had only 1/6 strict audit winners and is not inferential.

The route was predominantly atom-level (101/288 atoms versus 10/96 episodes).
Four fully reported posthoc cross-block grouping rules also failed to produce
a credible replacement: the exact-cell rule was flat against the student in
one direction, family and kind-level each lost a contrast, and family-kind's
positive reverse result selected deep only. Details and the full
machine-readable sensitivity are in
[route_diagnostics.md](route_diagnostics.md) and
`analysis/route_diagnostics.json`.

The independently regenerated 40/60 soup has weight hash `04610723…`; all 128
mapped adapter deltas were nonzero. On eight fixed same-prompt canaries, every
adapted arm changed from base, quick and deep differed 8/8, and soup differed
from quick 8/8 and deep 7/8. This rules out an installation/no-op failure but
is deliberately not scored as task evidence.

## Oracle Versus Deployable Boundary

The verifier and three-policy branch comparison are training-only acquisition
instruments. The primary deployable artifact must be one merged Qwen3.5-4B
checkpoint. The visible two-checkpoint router and verifier-best sample-more are
explicit baselines, not hidden components of the learned arm.

## Interpretation Contract

The first terminal branch is route nonexistence for the required *two-teacher*
composition claim. Locality, integration, controls, confirmation, and
benchmarks are unreached—not negative and not inferred.

## Best Next Test

Deep's replicated conditional advantage is the strongest surviving path. A
new experiment should first requalify that frozen deep route on fresh states
and test whether deep-only, verifier-backed MOPD can improve the existing soup
without erasing its quick behavior. That is the shortest clean test of the
still-untested update kernel.

Two-teacher composition should not reuse four-branch statewise argmax. It needs
cross-fitted direct estimates of each `teacher - student` advantage, sequential
branch allocation for uncertain states, a frozen predictor, and a third
untouched qualification block. If quick cannot independently replicate under
that design, it should be retired as a complementary teacher rather than
rescued with an observed-margin threshold. Any eventual checkpoint must still
beat both sources, the soup, visible routing, matched controls, and sample-more.

## Artifact Manifest

Large source, soup, adapter, and merged checkpoints are external under
`large_artifacts/qwen35_4b_same_prefix_advantage_routing/` as specified in
`artifact_manifest.yaml`. Small configs, receipts, raw score tables, analyses,
and narrative remain in git.
