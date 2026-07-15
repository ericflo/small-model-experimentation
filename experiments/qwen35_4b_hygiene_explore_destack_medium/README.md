# Hygiene-Explore De-stacked Dose with Medium Pilot

The minimal proven-install dose: only the two lessons that installed in every prior measurement (injection hygiene, budgeted route search), at their replicated per-kind dose, trained on the CLEAN surface-general parent — testing whether v2's failure was lineage saturation rather than content decay — with a gate that requires both installs to recover and the conditional pilot at the medium tier.

**Status:** in-progress · since 2026-07-15 · model-free construction under way; no model event has run

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

No model event has run.

## Interpretation

None yet.

## Knowledgebase Update

- Program evidence updated: pending.
- Program backlog updated: this trial claims queued direction (a).
- Claim ledger updated: no.

## Artifacts

- `data/sft_hygiene_explore.jsonl`, `data/corpus_manifest.json`: frozen corpus.
- `data/stream_manifest.json`, `data/stream_token_receipt.json`: exposure receipts.
- `data/local_tasks_seed88018.jsonl`, `data/local_input_seed88018.jsonl`, `data/local_design_receipt.json`: frozen gate.
- `reports/preregistration.md`, `reports/design_review.md`: contract and authorization.
- `reports/artifact_manifest.yaml`: external parent and conditional model-artifact plan.
