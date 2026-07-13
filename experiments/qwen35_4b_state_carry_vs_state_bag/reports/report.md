# State-Carry Versus State-Bag Counterfactual Report

## Status

`INVALIDATED_PILOT_RETRY_REQUIRED` / `SETUP_ONLY`: the first G0 and both fixed seed-7401 trainings
completed, but pilot analysis entered the full deployment comparator and rejected seed 7401 before
writing a verdict. The complete attempt is preserved as operational evidence only. The bug is fixed
and regression-tested; source binding requires fresh CPU/data/G0/pilot artifacts. No scientific LoRA
verdict exists yet.

## Question

Does a serially inherited internal state produce deeper, causally useful representations than an equal-compute collection of independent shallow states?

## Result

No valid result. The invalidated attempt recorded exact G0 parity and matched training receipts, while
both arms remained weak and Carry joint-state accuracy was chance-like. Because the registered
analyzer did not complete, those observations cannot promote, stop, or license the capacity successor.

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
