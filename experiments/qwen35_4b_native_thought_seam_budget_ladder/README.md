# Qwen3.5-4B Native-Thought Seam Budget Ladder

This study selects the smallest naturally closing Qwen3.5-4B thought cap from
256/512/1024 on fresh list tasks, then repeats that exact cap on untouched tasks
before any Jacobian value test.

## Research Program

- Primary: `interpretability_and_diagnostics`.
- Secondary: `test_time_reasoning_budget`.
- Cross-cutting catalog assignment: `benchmark_generalization`, because the
  repository's `ladder` tag routes frozen selection/confirmation ladders there.
- Program question: what exact natural reasoning interface is valid before a
  thought-prefix state can be assigned continuation value or causally edited?
- Direct parent: `qwen35_4b_native_thought_jacobian_value_transport`.
- Other anchors: `qwen35_4b_thinking_budget_scaling` and
  `qwen35_4b_answer_potential_trace_sft`.

## Question

On fresh, first-operation-identifiable list tasks, what is the smallest cap in
`[256, 512, 1024]` at which Qwen3.5-4B naturally emits `</think>` and a parseable
answer often enough to support a later thought-prefix value experiment—and does
that cap pass unchanged on an untouched confirmation split?

## Hypothesis

The failed 160-token parent exposed an interface-budget problem, not a J-space
result. Prior native-thinking evidence suggests that 512--1024 tokens is the
model's ordinary reasoning scale. At least one frozen rung should therefore
pass natural close, parse, headroom, and usable-prefix gates, then replicate on
fresh tasks. The smallest passing cap is selected; 1024 is expected to be the
most likely winner but receives no preferential rule.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Backend: Transformers bf16 SDPA, unpadded batch one, KV cache enabled. This is
  the backend required by the immediate activation-intervention successor.
- Task source: 40 newly generated procedural depth-two list transformations;
  16 budget-selection and 24 untouched confirmation tasks.
- Freshness: 40/40 unique fingerprints, zero overlap with both Jacobian parents;
  visible examples exhaustively identify one first-operation type.
- Sampling: temperature 0.6, top-p 0.95, top-k 20, three traces per task.
- Ladder: paired right-censoring at 256, 512, and 1024 thought-generation steps.
- Answer allowance: 16 naturally generated tokens after a natural close.
- Prohibited: injected close tokens, force-close answer generation, fallback to
  a different cap after confirmation, benchmark content, or correctness tuning.

The selection run generates each trace once to the 1024 ceiling and classifies
whether its natural close was reachable at each smaller cap. This is paired
right-censoring, not three independent samples. The untouched confirmation run
opens only the smallest selected cap.

## Frozen Decisions

Selection requires at least 80% natural close, 90% parsing conditional on close,
32 usable traces of at least 16 thought tokens, 5%--95% usable success, and six
mixed-success tasks. Confirmation scales the usable/mixed counts to 48/eight,
repeats the rates unchanged, and additionally requires the 95% Wilson lower
bound on natural close to be at least 0.70.

Terminal labels are `NO_BUDGET_SELECTED`, `SEAM_NOT_REPLICATED`, or
`NATURAL_SEAM_REPLICATED`. Only the last label licenses a separate value/Jacobian
experiment at the frozen selected cap. It is setup evidence, not a capability
gain or a J-space mechanism result.

## Run

CPU smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/qwen35_4b_native_thought_seam_budget_ladder/scripts/run.py \
  --stage smoke
```

After the immutable design boundary is anchored:

```bash
.venv/bin/python experiments/qwen35_4b_native_thought_seam_budget_ladder/scripts/run.py \
  --stage model-smoke
.venv/bin/python experiments/qwen35_4b_native_thought_seam_budget_ladder/scripts/run.py \
  --stage budget-selection
.venv/bin/python experiments/qwen35_4b_native_thought_seam_budget_ladder/scripts/run.py \
  --stage seam-confirmation
```

## Status

Design and adversarial review complete before any model call. CPU smoke passes:
40/40 fresh unique tasks, zero parent overlap, and every terminal gate is
mathematically reachable. Scientific outcomes remain unopened.

## Scope

This experiment selects an interface budget. It does not fit a value coordinate,
patch an activation, train a controller, compare capability against sampling, or
license a claim. A positive result merely fixes the natural seam for the next
result-bearing experiment, which must use exact-prefix replay and construct
post-bf16 controls at every live sequence length.

## Knowledgebase Update

- Program evidence: update after the terminal confirmation decision.
- Program backlog: records this as the required successor to the 160-token stop.
- Claim ledger: no claim ID; the repository claim re-grade remains open.

## Artifacts

- `data/procedural/`: frozen fresh splits and manifest.
- `runs/smoke/`: CPU and gate-reachability receipts.
- `runs/model_smoke/`: backend/token/cache plumbing only.
- `runs/budget_selection*`: paired ladder rows and frozen selection.
- `runs/seam_confirmation*`: untouched single-cap confirmation.
- `reports/preregistration.md` and `reports/design_review.md`: immutable rules.
- `reports/artifact_manifest.yaml`: omission and reproduction contract.
