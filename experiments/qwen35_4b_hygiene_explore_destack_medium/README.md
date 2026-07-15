# Hygiene-Explore De-stacked Dose with Medium Pilot

The minimal proven-install dose: only the two lessons that installed in every prior measurement (injection hygiene, budgeted route search), at their replicated per-kind dose, trained on the CLEAN surface-general parent — testing whether v2's failure was lineage saturation rather than content decay — with a gate that requires both installs to recover and the conditional pilot at the medium tier.

**Status:** finished · 2026-07-15 · not promoted (retention bands); RECOVERY CONFIRMED — both installs returned on the clean lineage (interference, not content decay), and the receipts isolate replay interleaving as the retention protector; seed 78,148 permanently sealed

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: the four-event installed-lesson map (hygiene 4/4 kind wins; explore most); the v2 kill-rule closure and third-dose interference law; the dose-two precedent (axis_on_replay installed cleanly with retention byte-equal).

## Question

Do the two replicated installs recover cleanly when de-stacked onto the clean lineage at dose two — adjudicating interference versus content decay — and does the medium tier then convert them?

## Hypothesis

Interference, not content decay, explains v2: the same lessons at the same per-kind dose on a twice-dosed-maximum lineage recover their installs. Falsified if either kind fails to strictly win its fresh holdout against both the parent and matched replay.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: `designed_fresh` composite (tree `93433aa2...255`, weights `0a3b89cd...979`); warm start from its adapter (`36f41095...442`).
- Corpus (construction seed 77,119): 80 rows — `u_hygiene` 40 (co-location-hardened v2 lesson), `u_explore` 40 (unchanged); executable truth; banned vocabulary audited.
- Arms: `replay_clean` (control) and `hygiene_explore` (candidate); 1,280-core + 240-block exact three-axis MILP (candidate block = 80 treatment + 160 fillers; slot seed 55,121); training seed 55; 190 updates; zero skips.
- Gate: fresh seed 88,018; 20-task two-kind holdout + 104-task retention; corrected detectability bar (with two kinds, BOTH must strictly win; ties fail; fail-closed if undetectable); retention bands unchanged; documented answer normalization; unconditional recovery flags in the receipt.
- Conditional pilot: sealed seed 78,148, MEDIUM tier, think budget 1,024; candidate aggregate strictly above base, replay control, and parent; every-family-versus-base recorded as the goal gate (8-of-92 historical medium passes).
- Hidden boundary: `benchmarks/` unread.

## Run

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_hygiene_explore_destack_medium/scripts/run.py --smoke
```

Checkpointed stages:

```bash
.venv/bin/python -B experiments/qwen35_4b_hygiene_explore_destack_medium/scripts/run.py --stage train-control
.venv/bin/python -B experiments/qwen35_4b_hygiene_explore_destack_medium/scripts/run.py --stage train-candidate
.venv/bin/python -B experiments/qwen35_4b_hygiene_explore_destack_medium/scripts/run.py --stage merge-arms
.venv/bin/python -B experiments/qwen35_4b_hygiene_explore_destack_medium/scripts/run.py --stage local
.venv/bin/python -B experiments/qwen35_4b_hygiene_explore_destack_medium/scripts/run.py --stage benchmark
```

## Results

Both arms trained cleanly (control 0.3831, candidate 0.4613 train loss; 0 skips) and merged. The frozen 124-task gate at seed 88,018 (normalized grading, both kinds detectable, both required to win): axis holdout of 20 — candidate 15, replay control 11, parent 8; per-kind candidate/parent/replay: explore 7/4/6 (win), hygiene 8/4/5 (win). RECOVERY FLAGS: `explore_win: true`, `hygiene_win: true` — the preregistered de-stacking reading is positive. Retention of 104: candidate 58/93/11 versus parent 68/98/7 and replay 66/86/19 — the candidate broke the correct band against both controls (−10/−8 vs the −5 band), the parent cap band (+4 vs +3), and the parent parse band (−5 vs −3). No promotion; seed 78,148 permanently sealed; the medium pilot never ran.

## Interpretation

Three frozen readings. (1) RECOVERY CONFIRMED: both replicated installs returned decisively on the clean lineage at matched exposure — v2's failure was lineage interference, not content decay; the escalation rule does not fire. (2) The retention cost isolates a mechanism: the one prior dose-two event that was retention-byte-equal (axis_on_replay) had a dedicated full replay round between doses; this trial dosed directly and paid ten retention points. Replay interleaving between designed doses protects retention — consistent with every replay-refresh observation in the line, and now localized to the dose boundary. (3) The gate did its job precisely: the axis instrument certified the installs while the retention bands correctly refused a candidate that forgets — this is the strongest axis result of the session (15/20, +4 over the best control) on a model that must not be deployed.

## Terminal Disposition

No later event is authorized here. Seed 78,148 is spent-by-sealing. The published composites and receipts are preserved; the `replay_clean` composite (retention 66/86/19 at the gate) and its adapter provide the interleaved-replay parent that the retention law points to for any successor dose, which requires its own intake and lifecycle.

## Knowledgebase Update

- Program evidence updated: recovery confirmation and the replay-interleaving retention law recorded.
- Program backlog updated: the interleaved-replay dose successor queued with calibration notes.
- Claim ledger updated: no.

## Artifacts

- `data/sft_hygiene_explore.jsonl`, `data/corpus_manifest.json`: frozen corpus.
- `data/stream_manifest.json`, `data/stream_token_receipt.json`: exposure receipts.
- `data/local_tasks_seed88018.jsonl`, `data/local_input_seed88018.jsonl`, `data/local_design_receipt.json`: frozen gate.
- `reports/preregistration.md`, `reports/design_review.md`: contract and authorization.
- `reports/artifact_manifest.yaml`: external parent and conditional model-artifact plan.
