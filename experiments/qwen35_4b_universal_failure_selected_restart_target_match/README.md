# Failure-Selected Counterfactual Restart Curriculum

**Status:** in-progress · since 2026-07-14 · parent rollout is preserved; model-free failure selection is next

This experiment tests whether selecting the stronger parent's fresh procedural
failures and teaching clean verified restarts can beat exactly exposure-matched replay
without conditioning on the parent's failed trajectory.

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can one Qwen3.5-4B checkpoint acquire execution, induction,
  verification, repair, state, and commitment behavior without trading away any
  held-out benchmark family?
- Prior anchors: C50's emission-seam installation result, C58's partial success from
  context-removal recomputation, and the terminal negative
  `qwen35_4b_universal_on_policy_prefix_repair_token_match`.

## Question

Does selecting tasks the deployed parent fails, then supervising a concise verified
solution from the original prompt, install broader competence than either continued
replay or conditioning the correction on the parent's long wrong prefix?

## Hypothesis

The predecessor solved on-policy data availability but put 47,123 wrong parent tokens
in the candidate's conditioning context and gave it 33,421 fewer supervised target
tokens than replay. A counterfactual restart moves the intervention before the error:
selection remains on-policy at the task level, while every trainable row begins at the
original prompt and contains a full truth-audited solution and answer. Exact matching
of forward tokens, loss-bearing target tokens, and absolute loss mass will isolate
that mechanism from extra compute or extra supervision.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: the published explicit `replay_after_close` composite, weight SHA-256
  `7ab4c419...36e2e`; runtime LoRA is forbidden.
- Collection substrate: 624 fresh procedural tasks, 48 per each of the 13 universal
  skills, construction seed 77,114. Truth source SHA-256 is `81edc9ea...de304`;
  oracle-free runner input SHA-256 is `25382689...0f5b`.
- Collection: one vLLM event, natural thinking, greedy `n=1`, seed 66,114,
  1,024-token cap, 4,096 context, and identical pinned runner geometry.
- Failure rule: cap contact, missing answer, wrong answer, or more than 128 thinking
  tokens. Hard correctness/cap failures rank before budget-only failures.
- Selection: exactly four failures per skill, 52 total, deterministic seed 55,114.
  An undersupplied skill ends the experiment before training.
- Planned control: same-parent replay continuation. Both future arms must have 320
  rows, 200 byte-identical aligned replay rows, 40 effective-batch-eight updates, and
  exact equality on forward tokens, loss-bearing targets, and absolute loss mass.
- Train seed: 48. Fresh local seed: 88,010. Conditional aggregate seed: 78,140.
- Hidden-label boundary: no benchmark item, transcript, source, or detailed result is
  read. Aggregate access remains sealed until strict local promotion.

## Run

The current checkpoint exposes the CPU smoke path and, after this collection result is
published green, model-free failure selection:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  experiments/qwen35_4b_universal_failure_selected_restart_target_match/scripts/run.py --smoke

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  experiments/qwen35_4b_universal_failure_selected_restart_target_match/scripts/run.py \
  --stage mine-restarts
```

No multi-stage invocation is supported. Training and evaluation remain unauthorized
until the observed restart source passes a second exact-exposure design review.

## Results

The one preregistered parent event completed from pushed-green commit `1744e753`:
624/624 completions, 304,013 sampled tokens, 879.9 tok/s, and 394.96 seconds of
wrapper wall time. Rollout/metadata/log/receipt SHA-256 values are
`4bf15134...1099f`, `b43b3a0...1206d`, `668e9b70...369ff`, and
`1d35c63a...2b381`. The receipt records a clean `main` preflight, the authenticated
merged replay parent, no recovery or generation rerun, `benchmark_data_read=false`,
and a sealed aggregate seed.

This is collection evidence, not a capability result. Failure composition and quota
availability remain unopened until the separately published mining stage.

## Interpretation

This trial distinguishes two ideas the predecessor confounded: selecting data from
the parent's actual failures, and conditioning training on the parent's failed
trajectory. A positive result would support task-level failure selection plus clean
recomputation, not long-prefix repair. A negative result would reject this balanced
13-skill restart package at the matched exposure, not all on-policy learning.

## Knowledgebase Update

- Program backlog: register as the active result-separated universal successor.
- Program evidence: update after the first preserved model event.
- Claim ledger: unchanged until held-out evidence warrants a claim.

## Artifacts

- `idea_intake.md` — closest duplicate and mechanism decision.
- `reports/preregistration.md` — frozen identities, gates, and checkpoint order.
- `reports/design_review.md` — adversarial pre-rollout review.
- `data/rollout_tasks_seed77114.jsonl` — executable truth and oracle restarts.
- `data/parent_rollout_input_seed66114.jsonl` — oracle-free vLLM input.
- `data/rollout_task_manifest.json` and `data/design_receipt.json` — freshness and design receipts.
- `src/vllm_runner.py` — pinned same-backend runner with explicit-composite gate.
