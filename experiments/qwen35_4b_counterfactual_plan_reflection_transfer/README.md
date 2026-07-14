# Qwen3.5-4B Counterfactual Plan Reflection Transfer

**Status:** in-progress · since 2026-07-14 · model-free construction smoke complete; adversarial design review required before any model or GPU event

This experiment tests the paper's most actionable claim without relying on its
consciousness framing: can supervision on what the model *would say if interrupted
and asked to reflect* change what it does on the unreflected action branch?

## Research Program

- Primary program: `posttraining_and_adaptation`.
- Cross-program fit: `structured_execution_and_compilers` and
  `interpretability_and_diagnostics`.
- Closest near-duplicate: `qwen35_4b_bank_the_thoughts`, which trains the actual
  plan-and-code continuation. Here the treatment receives loss only on a later,
  counterfactual reflection turn and never on the task answer.
- Other anchors: `qwen35_4b_commit_slot_semantic_power_replication` (shared scalar
  J-value negative), `qwen35_4b_jacobian_transport_control_replication` (clean
  supplied-concept J transport positive), and
  `qwen35_4b_tokenizer_eos_residual_mechanics_fresh_replay` (inference-time plan
  materialization did not create correct proposals).

## Question

Can correct, reflection-only SFT on fresh three-step machine-induction contexts
increase held-out answer coverage on the same contexts' unreflected action branch,
beating a within-family shuffled-reflection arm, frozen Qwen3.5-4B, and
matched-candidate sample-more?

## Hypothesis

An appended reflection question creates a training branch on which the model must
name the ordered latent plan but not calculate or state the query answer. If the
paper's verbal-disposition mechanism transfers to capability learning, gradients
from that final reflection answer should make the correct plan easier to assemble in
the shared pre-action context. The actual action answer is never a target for the
reflection arms. A gain is not evidence for this mechanism unless correct reflection
beats shuffled reflection under byte-identical contexts and matched training, and
then replicates on a fresh confirmation split against sample-more.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Task source: experiment-owned procedural list, string, and three-register
  machines. Every task composes exactly three parameter-free primitives, shows seven
  examples, and asks for outputs on three new inputs.
- Common context: the user supplies the machine and says not to solve; the Assistant
  gives the fixed content-free response `READY`. The next turn branches.
- Reflection branch: asks only for `PLAN: first -> second -> third`; its target never
  contains the exact query-answer string. Loss is restricted to that final Assistant
  turn.
- Action branch: asks only for `ANSWER: <JSON outputs>`. Reflection-only training
  never receives loss on this branch or its answer.
- Splits: proposed 288 train tasks and independent 144-task qualification and
  144-task confirmation blocks, balanced across three families; composition and
  behavioral signatures must be disjoint across all splits. Sizes remain
  design-reviewable and no model event is authorized yet.
- Baselines and controls: frozen action; frozen literal-reflection-then-action;
  correct reflection; within-family shuffled reflection; a direct
  plan-plus-answer SFT positive control; depth-1/2 retention; same-backend
  sample-more at candidate counts 1, 4, and 16.
- Primary deployable metric: paired exact query-answer accuracy and coverage at the
  same candidate count and generation cap. Report every family separately.
- Hidden-label boundary: answers are procedural oracle labels used only for grading
  and direct positive control construction. No `benchmarks/` path may be read,
  imported, or used for training.

## Staged Decision

1. CPU construction must prove exact re-execution, exact-depth feasibility, all
   identity/collision rules, shuffled-target derangement, and answer omission.
2. Adversarial design review must freeze transcript rendering, loss masks, token and
   optimizer matching, attainable gates, adapter seeds, and same-backend evaluation.
3. Frozen calibration must establish a parseable action interface and non-saturated
   headroom before training.
4. A one-seed qualification may open a second training seed only if correct
   reflection beats shuffled reflection and frozen sample-more on the untouched
   qualification split. The fresh confirmation block remains sealed until then.
5. Jacobian measurement and ablation remain sealed until the behavioral result
   passes on both training seeds and confirmation. The J stage must beat slot margin,
   logit/unembedding, equal-width non-J, and shuffled-label readouts before causal
   ablation; causal mediation then requires correct-operation ablation to erase more
   of the gain than wrong-operation and random same-norm controls.

No generic within-`<think>` correctness scalar is being retried: that exact proposal
was already tested and failed task-held-out controls in
`qwen35_4b_commit_slot_semantic_power_replication`.

## Run

Authorized model-free smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  experiments/qwen35_4b_counterfactual_plan_reflection_transfer/scripts/run.py --smoke
```

No full, model, training, evaluation, or Jacobian command is authorized yet.

## Results

The model-free smoke constructs 30 tasks (four/three/three per family across
train/qualification/confirmation), re-executes every target, and verifies zero task,
composition, or behavioral-signature collisions; the correct and shuffled arms have
byte-identical contexts and a strict within-family derangement. It makes zero model
calls, GPU events, or benchmark reads. This is readiness evidence only.

## Interpretation

The paper unlocks a training hypothesis, not an already-demonstrated Qwen capability.
Inference-time semantic materialization already failed in this repository, whereas
counterfactual reflection changes weights using loss on a different branch. The new
experiment exists to distinguish that mechanism from direct plan SFT and from mere
additional sampling. No scientific result exists yet.

## Knowledgebase Update

- Program evidence: unchanged until a model result exists.
- Program backlog: records this active reflection-only mechanism test.
- Claim ledger and shared synthesis: unchanged; no claim ID allocated.

## Artifacts

- `idea_intake.md`
- `configs/default.yaml`
- `src/taskgen.py`
- `src/vllm_runner.py`
- `scripts/run.py`
- `tests/test_taskgen.py`
- `tests/test_vllm_runner.py`
- `reports/artifact_manifest.yaml`
