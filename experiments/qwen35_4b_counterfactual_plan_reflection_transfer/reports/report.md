# Counterfactual Plan Reflection Transfer — Design-Hold Report

## Summary

The experiment remains without model forward passes under a full-implementation
adversarial HOLD. Full CPU construction and the historical tokenizer receipt succeed,
but Review-7 remediation invalidated that receipt as a training prerequisite. No Qwen
generation, GPU, training, capability measurement, or Jacobian event exists.

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
historical tokenizer receipt then establishes exact parity across correct reflection, shuffled
reflection, and auxiliary plan-label arms: 77,020 prompt tokens, 5,164 target tokens,
and 82,184 forward tokens each. All 12 correct/shuffled optimizer groups match. The
current implementation requires that evidence to be reissued with exact tokenizer,
worktree, and script commitments. This is construction/training-parity readiness
evidence only; it is not a model result.

## Controls

The repaired design now includes a rendered-token-matched non-reflective plan-label
arm, a direct action-branch positive control, real retention data, exact target-only
loss masks, within-optimizer-step derangement, a frozen QLoRA recipe, paired
qualification/confirmation gates, and a specified literal-reflection diagnostic.
The implementation now additionally binds exact base/tokenizer/runtime bytes across
training, merge, and post-vLLM-load boundaries, enforces a detached execution worktree,
and makes end-to-end matched-compute sample-more a transitive final-stage gate. Model,
GPU, training, evaluation, and J-space execution remain unauthorized pending Review 8.

## Oracle Versus Deployable Evidence

Procedural targets are oracle labels. A correct-reflection training target is not
deployable evidence. Only answer accuracy on the unreflected held-out action branch,
against frozen sample-more, can become deployable evidence.

## Interpretation

No capability inference is licensed. The complete Review-7 remediation passes 80
pinned-environment model-free tests, but this is implementation evidence only. A fresh
adversarial review of the exact pushed SHA remains blocking.

## Next Experiments

Push the complete Review-7 remediation, obtain a fresh Review-8 verdict, and remediate
any new counterexample before changing authorization. Nothing beyond tokenizer-only
work is authorized yet.

## Artifact Manifest

See `artifact_manifest.yaml`.
