# Counterfactual Plan Reflection Transfer — Design-Hold Report

## Summary

The experiment remains without model forward passes under a full-implementation
adversarial HOLD. Full CPU construction and the independently authorized tokenizer
receipt succeed, but no Qwen generation, GPU, training, capability measurement, or
Jacobian event exists.

## Research Program Fit

This is a posttraining experiment motivated by a mechanistic claim: supervise an
answer to a hypothetical reflection question and test whether the untrained action
branch improves. It is not another inference-time materialization prompt and not a
generic correctness probe.

## Method

Fresh exact-depth list, string, and register machines provide seven demonstrations
and three query inputs. Correct and shuffled arms share the same common transcript
and reflection question; only the final reflection answer differs. The reflection
names the three primitives and omits the exact final answer. Proposed deployment
asks for the answer on a different next-turn branch.

## Results

The full configured construction creates 576 unique exact-depth-three tasks: 216
train, 72 calibration, 144 qualification, and 144 confirmation, plus 48 exact-depth
1/2 retention tasks. It has zero cross-split program or behavior
collisions, unique exact plans on the seven visible demonstrations, complete
operation-position coverage, and behaviorally wrong shuffled donors. The pinned
tokenizer receipt then establishes exact parity across correct reflection, shuffled
reflection, and auxiliary plan-label arms: 77,020 prompt tokens, 5,164 target tokens,
and 82,184 forward tokens each. All 12 correct/shuffled optimizer groups match. This
is construction/training-parity readiness evidence only; it is not a model result.

## Controls

The repaired design now includes a rendered-token-matched non-reflective plan-label
arm, a direct action-branch positive control, real retention data, exact target-only
loss masks, within-optimizer-step derangement, a frozen QLoRA recipe, paired
qualification/confirmation gates, and a specified literal-reflection diagnostic.
Tokenizer parity is complete. Model, GPU, training, evaluation, and J-space execution
remain unauthorized pending remediation and another adversarial review.

## Oracle Versus Deployable Evidence

Procedural targets are oracle labels. A correct-reflection training target is not
deployable evidence. Only answer accuracy on the unreflected held-out action branch,
against frozen sample-more, can become deployable evidence.

## Interpretation

No capability inference is licensed. Review 3 passed only the tokenizer stage and
demonstrated false full-gate passes through forged labels, incomplete sampling seals,
and imbalanced family/depth bundles. Stage ancestry, literal reflection inputs,
adapter lineage/runtime parity, and live KV-capacity preflight also remain blocking.

## Next Experiments

Remediate all Review 3 blockers model-free and obtain a fresh adversarial verdict.
Nothing beyond that repair is authorized yet.

All eight blockers now have committed-code candidates and focused regression coverage,
including the exact forged-label, family/depth imbalance, stage-cardinality, sampling,
merged-lineage, and live-capacity seams. They remain non-authorizing until Review 4
inspects an immutable pushed revision.

## Artifact Manifest

See `artifact_manifest.yaml`.
