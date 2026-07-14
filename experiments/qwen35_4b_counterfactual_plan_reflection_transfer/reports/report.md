# Counterfactual Plan Reflection Transfer — Design-Hold Report

## Summary

The experiment remains without model forward passes under a full-implementation
adversarial HOLD. Full CPU construction succeeds and the historical tokenizer receipt
is invalid as a training prerequisite. Review 9's five load, counter, runtime,
environment, and selected-hardware gaps are now remediated model-free and await an
independent Review 10 of the exact pushed revision. No Qwen generation, GPU, training,
capability measurement, or Jacobian event exists.

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
The implementation binds exact base/tokenizer/runtime bytes across training, merge,
and protected load windows; enforces a detached execution worktree with no ignored
state; binds a hashed external isolated interpreter and exact GPU identity; reconstructs
generation compute from raw token arrays; and makes checkpoint-aware end-to-end
matched-compute sample-more a transitive final-stage gate. Model, GPU, training,
evaluation, and J-space execution remain unauthorized pending Review 10.

## Oracle Versus Deployable Evidence

Procedural targets are oracle labels. A correct-reflection training target is not
deployable evidence. Only answer accuracy on the unreflected held-out action branch,
against frozen sample-more, can become deployable evidence.

## Interpretation

No capability inference is licensed. The current implementation passes 90 local
pinned-environment model-free tests. It now authenticates content inside protected
load windows, reconstructs prompt and training spend from raw/sealed token evidence,
starts under `-I -B -S` with complete stage-specific environment-byte authentication,
documents the distinct training and vLLM runtimes, and binds the selected physical GPU
UUID. Those are implementation claims awaiting independent adversarial review.

## Next Experiments

Publish the exact remediated implementation and obtain a fresh independent Review 10
verdict before changing authorization. Nothing beyond tokenizer-only work is
authorized yet.

## Artifact Manifest

See `artifact_manifest.yaml`.
