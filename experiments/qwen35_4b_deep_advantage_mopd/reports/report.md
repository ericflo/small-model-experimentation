# Qwen3.5-4B Deep-Advantage MOPD Report

## Status

The preregistered CPU/scientific smoke, pinned-model preflight, installation
canary, fresh route qualification, and five-update exact-logit locality pilot
all pass. Four-round integration is authorized; no capability comparison has
run yet.

## Research Program Fit

This is the direct intervention follow-up to
`qwen35_4b_same_prefix_advantage_routing`. That experiment independently
qualified deep but stopped before MOPD because its separate quick prerequisite
failed. This new directory preserves that negative and tests only the surviving
deep mechanism.

## Method

Two fresh split-branch blocks must requalify the unchanged strict deep route on
the immutable 40/60 soup. If and only if that passes, a five-update exact-logit
pilot may authorize four rounds of deep-top-50 corrected reverse-KL MOPD with a
25% frozen-soup anchor. Matched controls move deep targets to non-advantage
states, replace deep with quick on the exact selected states, or imitate the
best deep continuation off-policy.

## Results

Preflight and installation checks pass. On the eight fixed canary prompts,
quick, deep, and soup each changed all 8 outputs relative to base; quick and
deep differed on 8/8, while soup differed from quick on 8/8 and deep on 7/8.
The source/soup gate therefore authorized fresh route qualification.

Qualification then passed. Across 384 new states and 9,216 same-prefix
continuations, the selector routed 54 states to deep (28/26 by block). Deep's
independent audit advantage over soup was `+0.16499`/`+0.12205`, pooled
`+0.14209` with one-sided 95% lower bound `+0.12297`. Against quick it was
`+0.20003`/`+0.14203`, pooled `+0.16910` with lower bound `+0.15337`. The
minimum support, both block signs, and both uncertainty gates passed.

The diagnostic quick route also passed on these fresh blocks: 29/18 routed
states, quick-over-soup `+0.08198`/`+0.17054` and pooled lower bound
`+0.10008`; quick-over-deep `+0.05378`/`+0.27883` and lower bound `+0.12129`.
This does not change the locked deep-only treatment. It strengthens the case
that a later two-teacher attempt should use the preregistered cross-fitted
direct-advantage predictor and a third untouched block rather than reusing
these outcomes.

The online locality round required three candidate batches. Across 576 failed
states, the selector found 90 deep routes; the assembler froze exactly 60 deep
capability units, 20 soup anchors, and 60 disjoint matched non-advantage
controls. Control matching was exact-cell for 57 units and family/kind for
three. The all-policy cache contains 140 samples, 35,147 active positions, and
hash-bound quick/deep/soup top-50 targets.

The locality pilot then consumed exactly 15 deep and five soup units over five
updates. Mean corrected top-50 loss was `0.05242`, below the `0.10` training
ceiling; held-probe loss improved `0.04773→0.02947` and top-50 overlap improved
`0.84840→0.85163`. On batch-of-one exact probes, centered non-target logit
drift was `0.02760`, relative entropy drop was `0.03112`, and target loss
improved `0.01293→0.01170`. All frozen checks passed and the machine-readable
authorization is `four_round_mopd`.

The exact locality measurement covers one midpoint active token for each of
the 20 consumed units, rather than every one of the 4,898 trained positions.
It therefore establishes the literal preregistered local-safety gate, not
global invariance. No procedural capability, control, sampling, routing, or
benchmark result exists yet.

## Oracle Versus Deployable Boundary

Same-prefix verifier scores are training-only acquisition evidence. The
deployable primary must be one explicit merged Qwen3.5-4B checkpoint with no
teacher, verifier, router, or tool hidden at inference time.

## Interpretation Contract

Route, locality, training, control, procedural confirmation, and blackbox
benchmark results are distinct stages. An unreached stage is not negative and
will not be inferred from an earlier stop.

## Artifact Manifest

Large source, soup, adapter, and merged checkpoints remain external under the
paths in `artifact_manifest.yaml`. Small configs, receipts, analyses, tests,
and reports remain in git.
