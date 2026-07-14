# Qwen3.5-4B Counterfactual Plan Reflection Transfer

**Status:** in-progress · since 2026-07-14 · adversarial HOLD; repaired full CPU construction passes, but no tokenizer/model/GPU/training event is authorized

This experiment tests the paper's most actionable claim without relying on its
consciousness framing: can supervision on what the model would say on a later
reflection branch change what it does on an unreflected action branch? The fixed
`READY` seam makes this controlled branch transfer, not a claim about a literal
interrupted internal action state.

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
matched-candidate sample-more? If it does, is the gain specific to reflection framing,
or does an equally sized ordinary auxiliary plan-label branch work just as well?

## Hypothesis

An appended reflection question creates a training branch on which the model must
name the ordered latent plan but not calculate or state the query answer. If the
paper's verbal-disposition mechanism transfers to capability learning, gradients
from that final reflection answer should make the correct plan easier to assemble in
the shared pre-action context. The actual action answer is never a target for the
reflection or auxiliary-label arms. A gain is not task-specific transfer unless
correct reflection beats shuffled reflection under byte-identical contexts and
stepwise token-matched training. It is not reflection-specific unless it also beats
the correct non-reflective auxiliary-label arm.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Task source: experiment-owned procedural list, string, and three-register
  machines. Every task composes exactly three parameter-free primitives, shows seven
  examples, and asks for outputs on three new inputs. The exact ordered plan is the
  only depth-three program consistent with the seven visible examples; globally
  behavior-equivalent spellings and shallower-equivalent programs are excluded.
- Common context: the user supplies the machine and says not to solve; the Assistant
  gives the fixed content-free response `READY`. The next turn branches.
- Reflection branch: asks only for `PLAN: first -> second -> third`; its target never
  contains the exact query-answer string. The target Assistant turn contains a short
  plan statement inside the Qwen thinking channel and the same plan in the final
  answer; prompt and fixed `READY` tokens are fully masked.
- Auxiliary-label branch: replaces only `Pause before solving.` with
  `Provide exact labels.` and uses the identical correct target. Its rendered prompt
  token count must exactly match the reflection branch for every row before training.
- Action branch: asks only for `ANSWER: <JSON outputs>`. Reflection-only training
  never receives loss on this branch or its answer.
- Splits: 216 train, 72 frozen calibration, 144 qualification, and 144 confirmation
  tasks, balanced across three families, plus 48 untouched depth-1/2 retention tasks.
  Programs and behavioral signatures are disjoint.
- Baselines and controls: frozen action; frozen literal-reflection-then-action;
  correct reflection; within-family shuffled reflection; correct non-reflective plan
  labeling; a direct action-branch plan-plus-answer SFT positive control; depth-1/2
  retention; and same-backend sample-more.
- Training parity: QLoRA rank 32/alpha 64/dropout 0.05 on all seven projection
  modules, three epochs, batch 1 × accumulation 18, 36 final-only optimizer steps.
  Every optimizer group contains six rows per family. Correct/shuffled derangement is
  restricted inside that group, so target and forward-token totals must match within
  every step, not merely in aggregate.
- Primary deployable metric: paired exact full-query coverage@16 under identical vLLM
  thinking/answer budgets. Candidate counts 1 and 4 are descriptive. Report every
  family separately.
- Hidden-label boundary: answers are procedural oracle labels used only for grading
  and direct positive control construction. No `benchmarks/` path may be read,
  imported, or used for training.

## Staged Decision

1. CPU construction must prove exact re-execution, exact-depth feasibility, all
   identity/collision rules, shuffled-target derangement, and answer omission.
2. A tokenizer-only receipt must prove exact rendering, mask boundaries,
   reflection/auxiliary prompt-length equality, and per-step correct/shuffled parity.
   It remains unauthorized until the repaired design passes adversarial review.
3. Frozen calibration must establish a parseable action interface and non-saturated
   headroom before training.
4. Screen seed 47 trains all four arms. The direct positive control must reach 0.50
   coverage@16 and improve over frozen by 0.20. Correct reflection must beat shuffled
   and frozen by at least 0.10 overall and 0.05 in every family, with paired-bootstrap
   lower bounds above zero.
5. Only that pass opens replication seed 53 for the three non-positive-control arms.
   Both seeds must independently pass qualification before the fresh confirmation
   split opens; both must independently pass confirmation. No seed selection or
   ensembling is permitted. Retention must remain within the frozen margins.
6. Reflection-specific interpretation additionally requires correct reflection to
   beat the non-reflective auxiliary arm by 0.05 with a positive paired lower bound.
   Otherwise any capability pass is generic auxiliary-plan transfer.
7. A replicated behavioral pass may open a **new, result-separated experiment** with
   fresh J-fit, J-confirmation, and causal-confirmation data. No J-space fitting or
   ablation may reuse this experiment's behavioral gates.

No generic within-`<think>` correctness scalar is being retried: that exact proposal
was already tested and failed task-held-out controls in
`qwen35_4b_commit_slot_semantic_power_replication`.

## Run

Authorized model-free smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  experiments/qwen35_4b_counterfactual_plan_reflection_transfer/scripts/run.py --smoke
```

The full configured CPU construction is also authorized:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  experiments/qwen35_4b_counterfactual_plan_reflection_transfer/scripts/run.py --construct
```

No tokenizer, model, GPU, training, evaluation, or Jacobian command is authorized yet.

## Results

The repaired full model-free construction deterministically creates 576 depth-three
tasks plus 48 depth-1/2 retention tasks. It has 576 unique ordered depth-three
programs and behavior signatures, zero cross-split collisions, unique visible exact
plans, complete operation-by-position support in every full split, and zero exact
answer strings in reflection targets. Shuffled supervision preserves
immutable task truth and uses a within-family donor plan that is observably wrong on
the recipient's demonstrations or queries. The construction also emits immutable
four-arm record and optimizer-schedule hashes. A Python audit hook denies file and
directory access beneath the repository benchmark root. This remains model-free and
does not lift the adversarial HOLD.

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
- `src/records.py`
- `src/scoring.py`
- `src/analyze.py`
- `src/vllm_runner.py`
- `scripts/run.py`
- `scripts/tokenizer_receipt.py`
- `scripts/train.py`
- `scripts/merge_adapter.py`
- `scripts/build_eval_inputs.py`
- `scripts/score.py`
- `scripts/analyze.py`
- `scripts/calibration_gate.py`
- `tests/test_taskgen.py`
- `tests/test_records.py`
- `tests/test_scoring.py`
- `tests/test_analyze.py`
- `tests/test_vllm_runner.py`
- `reports/artifact_manifest.yaml`
- `reports/power_analysis.md`
