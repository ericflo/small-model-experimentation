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

## Artifact Manifest

Large source, soup, adapter, and merged checkpoints are external under
`large_artifacts/qwen35_4b_same_prefix_advantage_routing/` as specified in
`artifact_manifest.yaml`. Small configs, receipts, raw score tables, analyses,
and narrative remain in git.
