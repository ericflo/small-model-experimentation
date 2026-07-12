# State-Carry Versus State-Bag Counterfactual Report

## Status

`G0_MECHANICS_PASS` / `SETUP_ONLY`: fresh CPU contracts and the complete source-bound corpus pass.
The pinned `Qwen/Qwen3.5-4B` live smoke also passes exact K=1 parity, both-arm gradient,
worst-format K=12, parameter-identity, and memory gates. G0 is explicitly non-scientific; no training
or outcome evaluation has occurred yet.

## Question

Does a serially inherited internal state produce deeper, causally useful representations than an equal-compute collection of independent shallow states?

## Result

Not run. The live G0 receipt records mechanics only (`scientific_evidence: false`): K=1 maximum-logit
error `0.0`, exact Carry/Bag K=1 equality, finite K=12, 11.21 GiB peak allocation, and identical
16,800,796-parameter trainable receipts. Do not infer a State-Carry advantage from these checks.

## Required Terminal Evidence

The eventual report must include:

- all three training seeds for continuous Carry and Bag;
- K-by-depth curves and exact crossed task×training-seed Carry-minus-Bag uncertainty;
- K=1 parity plus parameter and decoder-layer-token compute receipts;
- seed-paired initialization and cumulative training-compute hash equality;
- joint-state tracking trajectories and both query strata;
- identical-checkpoint edge-cut effects and bidirectional geometry-matched donor-state swaps with raw hashes;
- every held-out split;
- joint family+surface holdout robustness;
- three seed-matched explicit-CoT majority and oracle `pass@N` comparisons at matched compute, including crossed uncertainty, raw generations, and interface-validity gates;
- retention/overthinking results; and
- the exact fail-closed verdict emitted by `src/analysis.py`.

## Artifact Manifest

See `artifact_manifest.yaml`.
