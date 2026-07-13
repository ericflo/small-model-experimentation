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

## Controls

## Oracle Versus Deployable Evidence

## Interpretation

## Next Experiments

Implement and adversarially audit cache forking, one-shot layer hooks, live-bf16
control repair, fixed-cap sampler, and resource matcher. Only then may a pushed
boundary authorize label-free model mechanics.

## Artifact Manifest

See `artifact_manifest.yaml`.
