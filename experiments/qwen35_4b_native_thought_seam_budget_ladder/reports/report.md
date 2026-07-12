# Qwen3.5-4B Native-Thought Seam Budget Ladder Report

## Status

Design-frozen; no scientific model outcome has been opened.

## Purpose

The direct parent stopped because all 48 native-thinking traces were still
inside `<think>` at 160 tokens. This separate experiment selects and confirms a
natural-close cap before any continuation-value or causal J-space measurement.

## Method

See `preregistration.md`. The selection split uses paired right-censoring at
256/512/1024; the confirmation split opens only the smallest selected cap. No
close token is injected and no cap-bound trace receives an answer continuation.

## Results

Pending model smoke, budget selection, and eligible confirmation.

## Interpretation Boundary

Even `NATURAL_SEAM_REPLICATED` would be setup evidence only. It would license a
new thought-prefix value/Jacobian experiment, not establish value decodability,
causal transport, or deployable capability.

## Artifact Manifest

See `artifact_manifest.yaml`.
