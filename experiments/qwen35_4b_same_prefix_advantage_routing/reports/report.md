# Qwen3.5-4B Same-Prefix Advantage Routing Report

## Status

**Design locked and conditional implementation smoke passed; outcome-bearing
execution not yet started.** The next legal stage is the pinned model preflight,
followed by soup construction/canary and the split-branch route gate.

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

No task-model evidence exists yet. Forty-eight unit/invariant tests, all 14
procedural-family selftests, source hashes, frozen count geometry, and the full
fail-closed stage graph pass. These are engineering evidence only and cannot
support the hypothesis.

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
