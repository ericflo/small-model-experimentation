# Qwen3.5-4B Commit-Slot Jacobian Value Transport Report

## Status

Design-frozen; outcome-blind plumbing passes; no scientific outcome opened.

## Purpose

Natural close and close-only free-form output both failed before J value. This
experiment supplies fixed answer syntax but not identity, constrains the next
choice to public aliases, requires real thought to beat both an immediate slot
and the same thought tokens in shuffled order, and then tests whether a scalar J
value coordinate improves the semantic choice.

## Results

CPU smoke only: 96/96 unique fresh exact-depth tasks, zero overlap with four
parents, no visible depth-one fits, exact lens hash, and reachable gates. The
CPU check caught an initial generator-seed collision before design freeze; the
entire seed block was replaced and the final manifest has zero overlap.

After the immutable design commit, model smoke passed the pinned 32-layer,
2,560-wide revision; five rank-24 lens matrices; 12 distinct leading-space alias
tokens; fixed slot tokenization; finite constrained logits; and native/free-form
cache contracts. Peak allocated GPU memory was 8,515,461,632 bytes. The receipt
contains neither task correctness nor the chosen alias.

## Boundary

The slot and alias mask are deployment scaffolds. Any positive is constrained
choice evidence, not natural/free-form capability. Value and donor stages remain
oracle.

## Artifact Manifest

See `artifact_manifest.yaml`.
