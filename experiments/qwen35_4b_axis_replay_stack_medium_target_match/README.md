# Axis-on-Replay Stack with Medium Pilot

Stack the proven axis-atom install on the strongest replay-compounded parent (the 0.5081 composite), against a replay-squared control that simultaneously measures whether replay compounding continues at round two — with the conditional pilot at the MEDIUM tier, where the all-families goal is empirically reachable.

**Status:** in-progress · since 2026-07-14 · model-free construction under way; no model event has run

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: the goal-gap axis experiment (first local promotion; axis flipped warren, its replay control flipped rites and posted the line's best aggregate); three consecutive replay-compounding observations; 8-of-92 medium-tier all-families passes versus 1-of-65 at quick.

## Question

Do the axis install and replay compounding stack in one model without interference — and does the medium tier convert installed axis skills into family scores that quick-tier atoms miss?

## Hypothesis

The two effects have disjoint family footprints at seed 78,144 and different mechanisms (content install versus distribution refresh), so training the inherited axis corpus from the replay-compounded parent should preserve both; the replay-squared control separates compounding from content; medium-tier episodes reward the multi-turn-flavored axes the atoms install.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: the `replay_repeat` composite of the goal-gap axis experiment (weights `3df45004...0072`, tree `4c4f3561...acd7`); warm start from its adapter (`20be87b5...cd36`). Runtime LoRA forbidden.
- Treatment corpus: byte-identical inheritance of the frozen 160-row axis corpus (`e7a95d73...686e`; construction seed 77,117).
- Arms: `replay_squared` (exact-exposure replay continuation — the control AND the round-two compounding measurement) and `axis_on_replay` (the stack candidate); 1,280-core + 240-block streams, exact three-axis MILP, slot seed 55,119; training seed 53, 190 updates, zero skips.
- Local gate: fresh seed 88,015, the same two-instrument achievable design (40-task axis holdout + 104-task retention screen; relative wins, non-inferiority bands, no absolute per-kind floors), three composites (parent + both arms).
- Conditional pilot: sealed seed 78,145, MEDIUM tier, think budget 1,024, four weight-authenticated composites (base, parent, replay_squared, candidate). Gates: candidate aggregate strictly above base, replay_squared, and parent; the every-family-versus-base record is the goal gate at the tier where it has been passed 8 times in 92 events.
- Hidden boundary: `benchmarks/` unread; independent seeds and matched-compute sample-more remain mandatory before any universal claim.

## Run

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_axis_replay_stack_medium_target_match/scripts/run.py --smoke
```

Checkpointed stages (each requires its prerequisite committed at a clean, pushed, green `main`):

```bash
.venv/bin/python -B experiments/qwen35_4b_axis_replay_stack_medium_target_match/scripts/run.py --stage train-control
.venv/bin/python -B experiments/qwen35_4b_axis_replay_stack_medium_target_match/scripts/run.py --stage train-candidate
.venv/bin/python -B experiments/qwen35_4b_axis_replay_stack_medium_target_match/scripts/run.py --stage merge-arms
.venv/bin/python -B experiments/qwen35_4b_axis_replay_stack_medium_target_match/scripts/run.py --stage local
.venv/bin/python -B experiments/qwen35_4b_axis_replay_stack_medium_target_match/scripts/run.py --stage benchmark
```

## Results

No model event has run. Model-free construction is in progress.

## Interpretation

None yet.

## Knowledgebase Update

- Program evidence updated: pending the first result.
- Program backlog updated: this trial claims both queued directions (replay compounding; axis conversion at medium).
- Claim ledger updated: no.

## Artifacts

- `data/sft_axis160.jsonl`, `data/corpus_manifest.json`: inherited frozen corpus.
- `data/stream_manifest.json`, `data/stream_token_receipt.json`: exposure receipts.
- `data/local_tasks_seed88015.jsonl`, `data/local_input_seed88015.jsonl`, `data/local_design_receipt.json`: frozen gate.
- `reports/preregistration.md`, `reports/design_review.md`: prospective contract and authorization.
- `reports/artifact_manifest.yaml`: external parent and conditional model-artifact plan.
