# Counterfactual Plan Reflection Transfer — Design-Hold Report

## Summary

The experiment remains without model forward passes under a full-implementation
adversarial HOLD. Full CPU construction succeeds and the historical tokenizer receipt
is invalid as a training prerequisite. Review 9's five load, counter, runtime,
environment, and selected-hardware gaps were remediated model-free. Review 10 then
reproduced four narrower runtime/import/re-exec/device-auth blockers; all four now
have model-free fail-closed remediations awaiting exact-SHA Review 11. No Qwen
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
The implementation binds exact base/tokenizer/runtime bytes across training, merge,
and protected load windows; enforces a detached execution worktree with no ignored
state; binds a hashed external isolated interpreter and exact GPU identity; reconstructs
generation compute from raw token arrays; and makes checkpoint-aware end-to-end
matched-compute sample-more a transitive final-stage gate. Model, GPU, training,
evaluation, and J-space execution remain unauthorized after Review 10 returned HOLD.

## Oracle Versus Deployable Evidence

Procedural targets are oracle labels. A correct-reflection training target is not
deployable evidence. Only answer accuracy on the unreflected held-out action branch,
against frozen sample-more, can become deployable evidence.

## Interpretation

No capability inference is licensed. The current implementation passes the local
pinned-environment model-free tests. It now authenticates content inside protected
load windows, reconstructs prompt and training spend from raw/sealed token evidence,
starts under `-I -B -S` with complete stage-specific environment-byte authentication,
documents the distinct training and vLLM runtimes, and binds the selected physical GPU
UUID. Those implementation claims were then subjected to independent adversarial
review.
Review 10 accepted those Review-9 closures but rejected the remaining auth→import
window, unpinned interpreter/stdlib/native boundary, broken `-S` venv/re-exec semantics,
and PATH-resolved selected-device query.

The pending Review-11 implementation closes those paths by retaining an authenticated
inotify/lease/hash window across every artifact-relevant import, pinning the
interpreter and complete stdlib/system/CUDA/site surfaces, explicitly activating the
authenticated CUTLASS path under `-S`, removing adaptive Mamba re-exec, pinning the
absolute device-query binary, and binding its physical UUID row to CUDA's sole active
logical device after initialization. Real detached bootstrap/seal checks pass for
both stage-specific environments. This remains implementation evidence only.

## Next Experiments

Publish the exact remediation and obtain a fresh independent Review 11 verdict before
changing authorization. Nothing beyond tokenizer-only work is authorized yet.

## Artifact Manifest

See `artifact_manifest.yaml`.
