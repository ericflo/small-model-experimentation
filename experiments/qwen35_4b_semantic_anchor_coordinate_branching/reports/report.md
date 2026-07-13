# Qwen3.5-4B Semantic-Anchor Coordinate Branching Report

## Summary

Terminal `INVALID_MECHANICS_CONTROL` (parse). Continuation and all task-
correctness stages remained sealed.

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
prefixes are locked.

The one authorized mechanics run reproduces all controls after canonical
identity sorting: 880/880 numeric and 2,240/2,240 intervention rows pass, all
four prefixes match, and donor tensors remain immutable. The automatic result
is nevertheless `INVALID_MECHANICS_CONTROL` because unrestricted next-token
parse is 56/880 (6.36%) against 95%; consequence parse is 0/440. Consequence
full tops are always a formatting opener (`294` backticks and `146` opening
brackets).

Within the preregistered constrained token sets, text target and full donor each
write the direct alias on 43/44 rows, donor J on 42/44, and wrong-donor J writes
its own alias on 42/44 while the registered target is 0/44. The corresponding
computed-consequence results are 6/44, 6/44, 5/44, and 5/44 wrong-donor-own.
Donor J exceeds the worse non-J consequence rate by one row and lifts mean
conditional target probability by only `0.00170`.

Post-mechanics review found a second invalidity: both task maps advance by the
same cyclic index, so alias -> operation -> label is a single fixed 12-pair map
across all four tasks. Individual map rotation did not produce independent
composition. Thus task/label breadth could not have certified computation even
if the endpoint had passed.

## Controls

The preregistration freezes task-local alias meaning, intended result-label rotation,
live bf16 norm/span tolerances, source/donor position, no-alpha/no-layer-sweep
rules, and compute accounting including every donor capture.

Mechanics and calibration files differ in row order; the runner compares sorted
unique row objects, and those objects are exactly equal. Earlier audit wording
that promised raw-file byte equality was too strong and is corrected here.

## Oracle Versus Deployable Evidence

Mechanics is independent of task correctness but remains a write diagnostic.
Literal text is deployable. Activation arms are white-box interventions and must
beat literal text plus matched-compute sampling to count as elicitation.

## Interpretation

No continuation is authorized. The formal invalid verdict is driven by the
frozen parse contract, not numerical corruption. Counterfactually ignoring
parse would still stop at `ANCHOR_PROBE_UNREACHABLE`, because literal text and
full-state consequence rates are only 13.6%. This does not establish a clean
negative about native J consequence transport: the readout itself is
unreachable and the intended label-randomization control is broken. It does
show that the late opaque anchor provides no deployable advantage over literal
text in this setup.

## Next Experiments

A fresh experiment may move a concrete textual hypothesis before the reasoning
trajectory, generate full continuations, and compare visible-only selection
against matched-compute sampling. Do not repair this result-bearing experiment,
post-hoc accept formatting tokens, or sweep J layers/scales.

## Artifact Manifest

See `artifact_manifest.yaml`; all terminal receipts are committed and no
external artifact is required.
