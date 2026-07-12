# Qwen3.5-4B Forced-Commit Jacobian Value Transport Report

## Status

Design-frozen; outcome-blind model smoke passed; scientific outcomes unopened.

## Purpose

Natural closure failed on 48/48 traces through 1,024 tokens. This successor
treats close injection as the explicit deployed budget-controller action, first
validates that interface on fresh data, and only then asks whether forced-policy
continuation value is readable and causal in J space.

## Method

See `preregistration.md`. Selection tests 256/512/1024 paired prefixes and freezes
the smallest forced-commit policy meeting parse, headroom, mixed-task, and answer
termination gates. Untouched confirmation opens only that cap. Later value and
causal stages are strictly gated and currently refuse placeholders.

## Results

CPU smoke produced 96/96 unique fresh exact-depth-two tasks, zero overlap with
all three scientific parents, no visible depth-one fits, exact replicated lens
hash, and reachable seam gates.

The non-result-bearing model smoke verified the exact pinned model, special and
alias token IDs, and rank 24 at lens layers 4--8. Its native trace forward input
lengths were `[375, 1, 1, 1, 1, 1, 1, 1]`; forced replay appended eight thought
tokens plus close and used `[384, 1]`. Both cache audits passed. Close injection
was explicitly marked counterfactual and no correctness was computed or stored.
Scientific seam selection remains unopened.

## Interpretation Boundary

Injected close is counterfactual to natural termination. A positive seam proves
only that this explicit policy is usable. A positive causal result would remain
oracle mechanism evidence, not installed capability.

## Artifact Manifest

See `artifact_manifest.yaml`.
