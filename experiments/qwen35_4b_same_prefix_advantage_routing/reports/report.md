# Qwen3.5-4B Same-Prefix Advantage Routing Report

## Status

**Design locked; implementation smoke, pinned model preflight, and source/soup
installation all pass.** The split-branch route qualification is now the next
legal and unresolved scientific gate.

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

No task-model evidence exists yet. The unit/invariant suite, all 14 procedural
family selftests, source hashes, frozen count geometry, and full fail-closed
stage graph pass. Pinned vLLM semantic generation and an independent finite
Transformers/QLoRA-path forward pass also succeeded. These are engineering
evidence only and cannot support the hypothesis.

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

The report will preserve the first terminal branch: route nonexistence,
locality failure, integration/control failure, procedural pass with benchmark
failure, or full pass. Unreached stages will remain listed as unreached rather
than inferred.

## Artifact Manifest

Large source, soup, adapter, and merged checkpoints are external under
`large_artifacts/qwen35_4b_same_prefix_advantage_routing/` as specified in
`artifact_manifest.yaml`. Small configs, receipts, raw score tables, analyses,
and narrative remain in git.
