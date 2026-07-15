# Axis-on-Replay Stack with Medium Pilot

Stack the proven axis-atom install on the strongest replay-compounded parent (the 0.5081 composite), against a replay-squared control that simultaneously measures whether replay compounding continues at round two — with the conditional pilot at the MEDIUM tier, where the all-families goal is empirically reachable.

**Status:** finished · 2026-07-15 · local negative on the kind-breadth bar alone; axis install replicated on the new parent (+6 total, hygiene and tracefix wins, best termination); replay round two showed local drift; seed 78,145 permanently sealed

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

Both arms trained cleanly (control 0.3468, candidate 0.4547 train loss; 0 skips) and merged. The frozen 144-task gate event at seed 88,015: axis holdout of 40 — candidate 24, parent 18, replay_squared 15; per-kind candidate/parent/squared: explore 5/4/7, hygiene 9/5/5, protocol 8/8/3, tracefix 2/1/0. Retention of 104: candidate 64/98/6 (correct/parsed/caps), parent 65/92/12, squared 64/86/18. Nine of ten checks passed; the single failure was the 3-of-4 kind-breadth bar (hygiene and tracefix won; protocol tied at the parent's 8 — the second consecutive experiment where protocol ties at that ceiling; explore lost to the control's 7/10). No promotion; seed 78,145 permanently sealed; no benchmark event ran.

## Interpretation

Three frozen readings land clearly. (1) Stack survival: the axis install transfers across parents — total +6 over parent for the second time, hygiene at 9/10 twice, with the best termination in the event (6 caps versus 12 and 18) and retention inside every band. (2) Replay compounding: round two DRIFTED at the local instrument (parse 86, caps 18, axis 15/40 with wild kind variance including the 7/10 explore fluke that broke the breadth bar) — the aggregate compounding observed at seed 78,144 does not show up as local quality, so compounding is either aggregate-specific or seed-fortunate. (3) Instrument design: the protocol holdout has tied at the parent ceiling in two independent experiments (8/8/8 twice), making it undetectable and effectively converting the 3-of-4 bar into 3-of-3 on the remaining kinds — a measured design flaw for successors to fix prospectively, not a capability fact.

## Terminal Disposition

No later event is authorized here. Seed 78,145 was never opened and is spent-by-sealing. The published `axis_on_replay` composite carries the strongest local numbers of any artifact in the line (24/40 axis, 64/98/6 retention) and is the presumptive subject of a fresh-instrument re-adjudication with a detectability-corrected breadth bar; any such test is a new experiment with its own gate seed and lifecycle.

## Knowledgebase Update

- Program evidence updated: stack survival, replay round-two drift, and the ceiling-tie instrument flaw recorded.
- Program backlog updated: fresh-instrument re-adjudication of the published composites queued with calibration notes.
- Claim ledger updated: no.

## Artifacts

- `data/sft_axis160.jsonl`, `data/corpus_manifest.json`: inherited frozen corpus.
- `data/stream_manifest.json`, `data/stream_token_receipt.json`: exposure receipts.
- `data/local_tasks_seed88015.jsonl`, `data/local_input_seed88015.jsonl`, `data/local_design_receipt.json`: frozen gate.
- `reports/preregistration.md`, `reports/design_review.md`: prospective contract and authorization.
- `reports/artifact_manifest.yaml`: external parent and conditional model-artifact plan.
