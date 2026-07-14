# Counterfactual Plan Reflection Transfer — Design-Hold Report

## Summary

The experiment remains model-free under an adversarial HOLD. The repaired full CPU
construction succeeds, but no tokenizer, Qwen generation, training, capability
measurement, or Jacobian event exists.

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
operation-position coverage, and behaviorally wrong shuffled donors. See the README
and adversarial design review for the receipt and remaining blockers. This is CPU
readiness evidence only.

## Controls

The repaired design now includes a rendered-token-matched non-reflective plan-label
arm, a direct action-branch positive control, real retention data, exact target-only
loss masks, within-optimizer-step derangement, a frozen QLoRA recipe, paired
qualification/confirmation gates, and a specified literal-reflection diagnostic.
Tokenizer and model execution remain unauthorized pending adversarial re-review.

## Oracle Versus Deployable Evidence

Procedural targets are oracle labels. A correct-reflection training target is not
deployable evidence. Only answer accuracy on the unreflected held-out action branch,
against frozen sample-more, can become deployable evidence.

## Interpretation

No scientific inference is licensed. The experiment has repaired its construction
layer but has not passed design review.

## Next Experiments

None is authorized from this intake. First complete design and implementation review;
then follow the preregistered stop gates.

## Artifact Manifest

See `artifact_manifest.yaml`.
