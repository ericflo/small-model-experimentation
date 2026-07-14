# Counterfactual Plan Reflection Transfer — Intake Report

## Summary

The experiment is model-free and intake-only. A real construction smoke exists, but
no Qwen generation, training, capability measurement, or Jacobian event exists.

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

The smoke validates construction and arm identity without loading a tokenizer or
model. See the README for the current receipt. This is readiness evidence only.

## Controls

The design proposes frozen, within-family shuffled-reflection, direct-plan-answer,
literal-reflection-at-test, retention, and same-backend matched-sampling controls.
Exact training parity and gates remain subject to adversarial review.

## Oracle Versus Deployable Evidence

Procedural targets are oracle labels. A correct-reflection training target is not
deployable evidence. Only answer accuracy on the unreflected held-out action branch,
against frozen sample-more, can become deployable evidence.

## Interpretation

No scientific inference is licensed. The experiment has claimed a distinct question
and made the first model-free construction path real.

## Next Experiments

None is authorized from this intake. First complete design and implementation review;
then follow the preregistered stop gates.

## Artifact Manifest

See `artifact_manifest.yaml`.
