# State-Carry Versus State-Bag Counterfactual Report

## Status

`SETUP_ONLY`: the protocol and harness received a final adversarial pre-run revision; no result-bearing `Qwen/Qwen3.5-4B` call has occurred. The historical CPU-smoke receipt predates that revision, so fresh CPU contracts/data are required before G0.

## Question

Does a serially inherited internal state produce deeper, causally useful representations than an equal-compute collection of independent shallow states?

## Result

Not run. Do not infer evidence from CPU reference mechanics, compilation, or the future model smoke.

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
