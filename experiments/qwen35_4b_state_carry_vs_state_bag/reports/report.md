# State-Carry Versus State-Bag Counterfactual Report

## Status

`SETUP_ONLY`: code, CPU mechanics, protocol, and handoff documentation are complete; no result-bearing `Qwen/Qwen3.5-4B` call has occurred in this experiment.

## Question

Does a serially inherited internal state produce deeper, causally useful representations than an equal-compute collection of independent shallow states?

## Result

Not run. Do not infer evidence from CPU reference mechanics, compilation, or the future model smoke.

## Required Terminal Evidence

The eventual report must include:

- all three training seeds for continuous Carry and Bag;
- K-by-depth curves and exact paired Carry-minus-Bag uncertainty;
- K=1 parity and parameter/FLOP receipts;
- seed-paired initialization and cumulative training-compute hash equality;
- state-sufficiency trajectories;
- trained-checkpoint edge cuts and donor-state swaps;
- every held-out split;
- mixed-echo branch status;
- three seed-matched explicit-CoT majority and oracle `pass@N` comparisons at matched compute, including paired uncertainty;
- retention/overthinking results; and
- the exact fail-closed verdict emitted by `src/analysis.py`.

## Artifact Manifest

See `artifact_manifest.yaml`.
