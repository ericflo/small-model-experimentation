# Qwen3.5-4B Commit-Slot Semantic Power Replication Report

## Status

Design-frozen; CPU, power, and outcome-blind model plumbing pass; no scientific
outcome opened.

## Purpose

The parent fixed slot repaired answer mode and showed a +8.33pp ordered-over-
shuffled hint at cap 1,024, but it missed the mixed-task gate and task-level
uncertainty crossed zero. This experiment fixes that one cap/interface and tests
the hint with 113 fresh task units per seam stage plus bootstrap and diversity
gates.

## Results

CPU only: 322/322 unique exact-depth tasks, zero overlap with five parents,
balanced 10--11 target tasks per seam split, exact lens hash, reachable gates,
and seven passing unit tests. Parent-effect planning requires and assigns 113
tasks per stage for approximate power 0.802745; the actual decision uses a
10,000-resample task bootstrap.

No correctness row, scientific trace, or scientific summary exists yet.

After the immutable design boundary, outcome-blind model smoke passed the exact
revision, 32-layer/2,560-wide architecture, five rank-24 lens matrices, 12
distinct leading-space aliases, fixed slot tokens `[271, 5170, 25]`, finite
logits, and native/free-form cache contracts. Peak allocation was 8,514,319,872
bytes. The receipt stores no correctness, chosen alias, or trace text. A final
implementation audit then verified task bootstrap, diversity, exact cardinality,
and confirmation hash locks before any scientific run.

## Boundary

Even a replicated pass is constrained semantic elicitation, not J certainty or
installed capability. Later J/value/control/causal commands fail closed.

## Artifact Manifest

See `artifact_manifest.yaml`.
