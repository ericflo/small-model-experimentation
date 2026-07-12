# Qwen3.5-4B Commit-Slot Semantic Power Replication Report

## Status

Design-frozen; CPU and power receipts pass; no model outcome opened.

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

No Qwen call, correctness row, trace, or scientific summary exists yet.

## Boundary

Even a replicated pass is constrained semantic elicitation, not J certainty or
installed capability. Later J/value/control/causal commands fail closed.

## Artifact Manifest

See `artifact_manifest.yaml`.
