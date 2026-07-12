# Preregistration: Prospective Prefix J-Value

Frozen after the automatic `POWERED_COMMIT_SLOT_SEAM_REPLICATED` decision and
before opening the reserved `value_fit` split or making a new model call. The
original seam rules and result cannot change. This document defines the newly
licensed measurement stage only; `causal_confirmation` remains sealed.

## 1. Question

At the live midpoint and endpoint of a sampled cap-1,024 native thought, do the
frozen five-layer J-coordinates rank the path's later semantic commit value on a
new task better than ordinary/direct readouts?

The load-bearing prospective question is whether coordinates at fraction 0.5
predict the deterministic correct-alias probability at the final fraction-1.0
commit. Endpoint coordinates are included to localize where value becomes
readable, but endpoint signal alone cannot pass the stage.

## 2. License and data firewall

- Require committed selection decision `POWERED_COMMIT_SLOT_SEAM_QUALIFIED`
  and independent confirmation decision `POWERED_COMMIT_SLOT_SEAM_REPLICATED`.
- Verify every selection/confirmation raw hash and the original data manifest.
- Open only 48 `value_fit` tasks. Do not read or generate from the 48
  `causal_confirmation` tasks during implementation, smoke, fitting, feature
  selection, thresholding, or interpretation.
- Preserve the exact Qwen/Qwen3.5-4B revision, Transformers bf16 SDPA backend,
  cap 1,024, three traces/task, sampler, prompt, aliases, close token, and slot.
- The value stage gets one scientific run. Interruption preserves raw failure;
  no same-experiment seed, fold, layer, label, ridge, or threshold retry.

## 3. Trace and prefix construction

Sample three native paths per value task to the fixed 1,024-token thought cap.
Retain thought before a natural close if one occurs; malformed pre-close EOS is
an unusable row in the denominator. For each usable path, construct fractions
0.5 and 1.0 of its available thought, with at least 16 tokens. Record the exact
prefix length and token hash.

The complete contract is 144 audited traces and 288 finite prefix rows, with
trace indices 0/1/2 present in every task/fraction group. The inherited 192-row
gate remains recorded, but missing/malformed paths make the stage
`INVALID_PREFIX_VALUE` rather than silently changing the candidate pool.

The feature position is the final included thought token under one cache-free
forward containing only prompt plus that live thought prefix—no injected close,
`First:` slot, answer token, or later thought. This prevents future-context and
answer-side activations from entering the J feature.

For each same prefix, separately append the registered close plus `First:` and
record ordinary constrained slot margin. This answer-side baseline never enters
the J feature forward.

At the full path only, append the same close/slot and record the constrained
probability assigned to the task's correct alias. That deterministic scalar is
the terminal value label for both the half and full prefix of that path. Thus
the half-prefix target is genuinely downstream along the sampled continuation;
the full-prefix target is seam-adjacent localization.

## 4. Frozen J feature

Load the exact replicated 24-coordinate lens, SHA-256
`e373b6e93956fdfc5cb446e9bee8249655707c8258a7868f0653d11f1ffd0213`.
At block outputs 4, 5, 6, 7, and 8, read all 24 pseudoinverse coordinates from
the selected residual, using rtol `1e-5`, and concatenate in layer-major frozen
concept order for 120 features. No layer/coordinate selection occurs.

Construct a second fixed 24-dimensional dictionary per layer from the
outcome-blind seed `2026071515`: project Gaussian columns out of the complete J
span, orthonormalize, and require maximum projection back into J-space <=1e-5.
The resulting 120 layer-matched non-J coordinates are a generic residual-state
baseline, never an intervention control.

All dictionaries must have effective rank 24 and every coordinate must be
finite. The feature schema, layer order, concept order, residual position, and
lens hash are runtime assertions.

## 5. Task-relative estimand

The intended use is ranking multiple sampled prefixes for one task, not
predicting which aliases are globally easy. Within each `(task, fraction)` group
of three paths, subtract the group feature mean and group terminal-label mean.
This transformation is label-free for features and available to a candidate
selector; label centering is training/evaluation only.

Assign whole tasks—not paths or prefixes—to four deterministic, alias-stratified
folds. All fractions and paths of a task remain in one fold. For each fold:

1. fit feature standardization on training-task rows only;
2. fit one fixed-L2 ridge model on the centered training labels;
3. transform the held-out task with training statistics; and
4. store one out-of-fold score per prefix.

No held-out labels affect centering, scaling, coefficients, coordinate choice,
regularization, or sign.

## 6. Pairwise task-macro score

Within each task/fraction, compare every path pair whose terminal correct-alias
probabilities differ by at least 0.01. A prediction earns 1 for the right order,
0 for the wrong order, and 0.5 for a tied score. Average eligible comparisons
within each task, then average task scores. This is called task-macro pairwise
AUC; no path-level ROC or pooled row count substitutes for it.

`mixed_tasks` counts tasks with at least one eligible terminal-value pair.
`scored_prefixes` counts finite out-of-fold rows, not pair comparisons.

## 7. Mandatory baselines

Run the identical folds, centering, standardization, ridge, and pairwise metric
for:

1. **correct-alias activity:** the five J coordinates corresponding to the gold
   alias, an explicitly oracle direct-activity baseline;
2. **ordinary slot margin:** top-one minus top-two constrained alias logit at the
   same forced prefix, an answer-side label-free confidence baseline; and
3. **alias identity:** a gold-alias one-hot baseline, included to expose global
   target priors. Within-task centering should make it exactly non-ranking; and
4. **non-J random coordinates:** 120 layer-matched coordinates from the frozen
   random subspaces orthogonal to J-space, testing whether equal-width generic
   early residual state explains the same value.

The primary 120-coordinate score must exceed all four. Gold probability or
gold logit from the prefix slot is the outcome/direct-tautology diagnostic and
is never a competitor the J model can claim to beat.

## 8. Shuffled null

For 32 frozen repeats, independently permute complete 120-coordinate vectors
among the three paths within every task/fraction, refit the same task-held-out
pipeline, and recompute task-macro AUC. This preserves task, fraction, alias,
feature distribution, and candidate count while breaking path-value alignment.
Gate the mean null AUC's absolute distance from 0.5.

## 9. Uncertainty

Use 10,000 stable task bootstrap resamples. Compute one-sided 95% lower bounds
for primary AUC above chance and for paired per-task primary-minus-correct-
activity and primary-minus-slot-margin differences. Prefixes and path pairs are
never resampled as independent observations.

## 10. Frozen pass gate

All must hold:

- at least 24 mixed tasks;
- at least 192 finite scored prefix rows;
- overall task-macro pairwise AUC at least 0.65;
- half-prefix prospective task-macro pairwise AUC at least 0.58;
- primary minus correct-alias activity at least 0.03;
- primary minus ordinary slot margin at least 0.02;
- primary minus alias identity at least 0.10;
- primary minus layer-matched non-J random coordinates at least 0.02;
- one-sided task-bootstrap lower primary AUC above 0.50;
- one-sided task-bootstrap lower primary-minus-correct-activity above 0;
- one-sided task-bootstrap lower primary-minus-slot-margin above 0;
- one-sided task-bootstrap lower primary-minus-non-J-random above 0;
- mean shuffled-null AUC within 0.05 of 0.50; and
- finite feature-row rate exactly 1.0.

Pass is `PREFIX_J_VALUE_PASS`; an intact run missing any scientific bar is
`NO_PREFIX_J_VALUE`; any hash, cardinality, schema, rank, leakage, finite, fold,
or outcome-timing contract failure is `INVALID_PREFIX_VALUE`.

No subgroup, pooled stage, alternative label transform, layer subset, sign
flip, ridge sweep, probability threshold, or post-hoc decoder can rescue a miss.

## 11. Interpretation

A pass means the frozen J representation contains task-relative prospective
information about a path's later constrained commit beyond the strongest
registered direct/activity baselines. It does not prove calibrated certainty,
token-local causality, autonomous termination, free-form answering, or installed
capability.

A pass licenses separate exact-control implementation and calibration. A later
causal stage remains oracle unless donor/axis choice becomes label-free. A
deployable capability claim still requires a new controller on untouched tasks
that beats frozen inference and matched-compute sampling under one backend.
