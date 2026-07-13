# Qwen3.5-4B Semantic-Anchor Coordinate Branching Report

## Summary

No model result exists. The experiment is at its immutable-design boundary.

## Research Program Fit

Primary `interpretability_and_diagnostics`; conditional secondary evidence is
withheld until the respective mechanism/capability gates run.

## Method

One shared 512-token native thought receives a valid source-alias anchor. Clean
target text is compared with full activation, context-local all-24 donor
coordinates, mean donor coordinates, the prior additive J method, two exact live
non-J controls, wrong donor, and logit-lens replacement. A direct identity probe
and a task-randomized computed operation consequence precede continuation.

## Results

Model-free smoke passes: lens SHA/rank and all 12 distinct diagnostic results
are valid; the 4/24/48 splits contain 76 unique new behaviors with zero overlap
against 1,046 ancestor fingerprints. The original smoke had six tests; 15 now
passed after the pre-model implementation audit (16 after the suffix-shape
repair). At that original boundary, model/outcomes remained unloaded.

Outcome-blind model smoke now also passes after preserving one failed receipt.
The equal-length retry has exact zero cross-suffix anchor difference, 20/20 live
non-J controls, maximum norm error `9.3031e-6`, maximum J-span fraction
`0.0099543`, and 60/60 intervention rows. No outcome has been measured.

The full outcome-blind calibration also passes all 880 live controls and 2,240
intervention rows. Maximum norm error is `9.9968e-6`, maximum span fraction is
`0.0099951`, and at most three lattice pairs are required. All four exact native
prefixes are locked. Mechanics probabilities remain unopened.

## Controls

The preregistration freezes task-local alias meaning, result-label rotation,
live bf16 norm/span tolerances, source/donor position, no-alpha/no-layer-sweep
rules, and compute accounting including every donor capture.

## Oracle Versus Deployable Evidence

Mechanics is independent of task correctness but remains a write diagnostic.
Literal text is deployable. Activation arms are white-box interventions and must
beat literal text plus matched-compute sampling to count as elicitation.

## Interpretation

Pending the staged decisions.

## Next Experiments

None authorized before mechanics. Failure of the explicit-anchor donor clamp
retires native J branching rather than opening another token/layer/scale search.

## Artifact Manifest

See `artifact_manifest.yaml`; no external artifact is currently required.
