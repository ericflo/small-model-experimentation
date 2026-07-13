# Qwen3.5-4B Jacobian Counterfactual Branching Report

## Summary

Design/CPU smoke only. No model or correctness outcome has run.

## Research Program Fit

## Method

The preregistration freezes zero-sum centered semantic branches, exact-Gram
J-orthogonal controls, cache-fork generation, full-path sample-more resource
matching, and a label-free mechanics gate. See the 22-point design review.

## Results

CPU/data smoke only:

- 76 fresh exact-depth-two task fingerprints are unique and disjoint from 634
  direct-ancestor fingerprints;
- lens hash is exact and layers 4--8 retain rank 24;
- every alpha/layer J branch bank is width 12, rank 11, and zero-sum within
  `5.22e-7` maximum coordinate residue;
- non-J branch Gram relative error is at most `1.14e-6`;
- float non-J projection into the complete J span is at most `3.05e-7`; and
- no model, outcome, correct alias, or confirmation stage was opened.

The fifth outcome-blind live smoke subsequently passes all 60 post-bf16 non-J
controls. Maximum paired norm error is `9.3881e-6`, maximum complete-J-span
projection `0.00912094`, and lattice repair uses at most five pairs. It records
no behavioral or target-selection metric.

Label-free mechanics is terminal `NO_NATIVE_J_BRANCH_CONTROL`:

| alpha | J target selected | non-J | mean J target-probability lift | numeric |
| ---: | ---: | ---: | ---: | --- |
| 0.5 | 4/48 | 4/48 | +0.000471 | pass |
| 1.0 | 4/48 | 4/48 | +0.001498 | pass |
| 2.0 | 4/48 | 4/48 | +0.005664 | pass |

Every write is finite and every live control passes, but target selection is
exactly the 1/12 chance rate with zero J specificity. No alpha meets the 60%,
+0.15 lift, or +35pp J-minus-non-J gates.

## Controls

## Oracle Versus Deployable Evidence

## Interpretation

The causal lens does not transfer through centered additive directions at an
arbitrary last-thought token. This does not contradict donor-coordinate
transport at an explicit semantic token. It isolates the next mechanism test to
semantic anchoring and coordinate replacement; continuation branching here is
cancelled.

## Next Experiments

Implement and adversarially audit cache forking, one-shot layer hooks, live-bf16
control repair, fixed-cap sampler, and resource matcher. Only then may a pushed
boundary authorize label-free model mechanics.

Superseded by the terminal mechanics gate. Create a distinct successor for an
explicit hypothesis anchor and donor-coordinate replacement; do not implement
cache-fork continuations in this experiment.

## Artifact Manifest

See `artifact_manifest.yaml`.
