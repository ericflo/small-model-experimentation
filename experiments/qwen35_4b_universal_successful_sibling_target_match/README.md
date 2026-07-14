# Policy-Supported Successful-Sibling Universal Curriculum

Test whether the deployed parent’s own short verifier-correct sampled trajectories can turn fresh greedy failures into transferable supervision under exact-exposure replay.

**Status:** finished · 2026-07-14 · terminal greedy-failure availability stop; sibling sampling, training, and benchmark access never opened

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can synthetic training data install a general feature that improves the held-out aggregate without regressing any reported family?
- Prior anchors: C54's shortest-success compression signal; the terminal on-policy-prefix negative; the terminal clean-restart negative.

## Question

Does the prior local failure come from teaching hand-authored trajectories outside the deployed parent’s policy support? This trial keeps the fresh greedy-failure selector but replaces every oracle restart with a short successful trajectory sampled from the same authenticated parent.

## Hypothesis

On greedy-failure tasks, a verifier-correct sibling proves that the complete reasoning and answer path already lies inside the parent’s sampling support. Distilling the shortest such path from the original prompt should bank a reachable decision policy rather than merely teach answer emission, and should therefore beat an independently trained exact-exposure replay continuation.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, pinned at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: authenticated explicit composite `replay_after_close` from `qwen35_4b_universal_on_policy_prefix_repair_token_match`; runtime LoRA is forbidden.
- Task source: 624 fresh executable procedural tasks, 48 per each of 13 universal skills, construction seed 77,115.
- Collection: one natural-thinking greedy event at seed 66,115; then, only for its published hard failures, one natural-thinking `n=16` event at seed 66,116 with temperature/top-p/top-k `0.6/0.95/20`.
- Candidate source: four tasks per skill whose sampled sibling naturally stops, closes thinking, exactly matches executable truth, uses a canonical answer tail, and stays within 768 thinking tokens. The shortest qualified sibling wins. There is no oracle-trace fallback.
- Baseline: unchanged authenticated parent.
- Active control: independent same-parent replay continuation, later matched exactly on forward tokens, loss-bearing target tokens, absolute loss mass, updates, and aligned replay rows.
- Fresh local gate: unchanged 26-task, two-per-skill gate at seed 88,011; strict total and execute/induct/probe wins over both parent and replay are mandatory.
- Hidden-label boundary: `benchmarks/` remains unread. Aggregate seed 78,141 stays sealed until local promotion.
- Final claim boundary: even a strict all-family aggregate pilot still owes independent higher-tier confirmation and a matched-compute sample-more baseline.

## Run

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_universal_successful_sibling_target_match/scripts/run.py --smoke
```

Checkpointed stages:

```bash
.venv/bin/python -B experiments/qwen35_4b_universal_successful_sibling_target_match/scripts/run.py --stage collect-greedy
.venv/bin/python -B experiments/qwen35_4b_universal_successful_sibling_target_match/scripts/run.py --stage prepare-siblings
.venv/bin/python -B experiments/qwen35_4b_universal_successful_sibling_target_match/scripts/run.py --stage collect-siblings
.venv/bin/python -B experiments/qwen35_4b_universal_successful_sibling_target_match/scripts/run.py --stage select-siblings
```

Every stage requires its prerequisite receipt committed on clean, synchronized `main`. Each result is checked, rebased, pushed, and held until both repository workflows are green before the next event.

## Results

Model-free construction and the sole greedy event completed. The authenticated parent produced 624/624 rows and 296,259 sampled tokens at 859.6 tok/s with no recovery or rerun. The frozen failure gate found 227 hard failures overall, but per-skill availability was `count=0`, `route=0`, and `select=2`, below the mandatory four for each skill. The outcome is `STOP_INSUFFICIENT_GREEDY_FAILURES`; inventory/selection-receipt hashes are `8e21caf8...d783` / `3397b773...2a6e`. No sibling input was emitted, sibling seed 66,116 was not consumed, and training, local evaluation, and benchmark access never opened.

## Interpretation

This is a prerequisite negative, not a test of successful-sibling distillation. A balanced failure-only curriculum is ill-posed when the deployed parent already clears some procedural skills greedily: count and route supply no failures, while select supplies only two. The useful result is to separate residual repair from retention. A successor may reuse this published failure pool in a new directory, sample only the ten skills with at least four failures, and use exact-exposure replay plus the unchanged all-skill local gate to protect mastered skills.

## Knowledgebase Update

- Program evidence updated: terminal availability result recorded.
- Program backlog updated: residual-only successor is queued; this record is closed.
- Claim ledger updated: no; no capability result exists.

## Artifacts

- `data/collection_tasks_seed77115.jsonl`: executable truth, never passed whole to the model.
- `data/greedy_input_seed66115.jsonl`: oracle-free model input.
- `data/collection_task_manifest.json` and `data/design_receipt.json`: frozen provenance.
- `reports/preregistration.md` and `reports/design_review.md`: prospective contract and adversarial authorization.
- `reports/artifact_manifest.yaml`: external parent and future adapter/composite plan.

## Terminal Disposition

No later event is authorized here. Do not lower the quota, borrow across skills, sample the two select failures, or add oracle rows. Any residual-skill design is a new experiment with its own intake, receipts, and lifecycle.
