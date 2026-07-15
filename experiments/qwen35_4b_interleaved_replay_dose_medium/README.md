# Interleaved-Replay Dose with Medium Pilot

The direct test of the replay-interleaving retention law: the same verified two-lesson dose that just recovered its installs (but paid ten retention points when dosed directly), now warm-started from the already-built interleaving replay round — reproducing the only retention-safe dose recipe the line has measured, with the conditional pilot at the medium tier.

**Status:** in-progress · since 2026-07-15 · model-free construction complete and smoke-green; no model event has run

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: recovery confirmed (both flags TRUE, axis 15/20 vs 11/8); the replay-interleaving law isolated across receipts; the interleaving parent already trained, merged, and receipted.

## Question

Does replay at the dose boundary protect retention while the proven installs land — completing the recipe for a deployable dosed model — and does the medium tier then convert it?

## Hypothesis

Replay re-anchors the retained-skill distribution before the dose perturbs it. Falsified if the dosed candidate still breaks the retention bands or the installs fail to win from the interleaved parent.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: the de-stack experiment's `replay_clean` composite (tree `19759e12...f67`, weights `2cef3e5e...0b4`); warm start from its adapter (`f6f910ed...bb8`).
- Corpus: byte-identical inheritance of the verified 80-row hygiene+explore corpus (construction seed 77,119).
- Arms: `replay_interleaved2` (control) and `dose_after_replay` (candidate); identical stream geometry; slot seed 55,122; training seed 56.
- Gate: fresh seed 88,019; two-kind holdout (both must win) + 104-task retention with unchanged bands; normalization; unconditional recovery flags; escalation rule frozen (retention breaking despite interleaving escalates to a mechanism study).
- Conditional pilot: sealed seed 78,149, MEDIUM tier, think budget 1,024; candidate aggregate strictly above base, control, and parent; every-family-versus-base recorded as the goal gate.
- Hidden boundary: `benchmarks/` unread.

## Run

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_interleaved_replay_dose_medium/scripts/run.py --smoke
```

Checkpointed stages:

```bash
.venv/bin/python -B experiments/qwen35_4b_interleaved_replay_dose_medium/scripts/run.py --stage train-control
.venv/bin/python -B experiments/qwen35_4b_interleaved_replay_dose_medium/scripts/run.py --stage train-candidate
.venv/bin/python -B experiments/qwen35_4b_interleaved_replay_dose_medium/scripts/run.py --stage merge-arms
.venv/bin/python -B experiments/qwen35_4b_interleaved_replay_dose_medium/scripts/run.py --stage local
.venv/bin/python -B experiments/qwen35_4b_interleaved_replay_dose_medium/scripts/run.py --stage benchmark
```

## Results

No model event has run. Model-free construction is complete: the inherited corpus authenticated by byte-identical re-derivation; the exact three-axis MILP solved optimally (both 240-row variable blocks at forward 147,792 / nonzero 63,001 / mass×5 71,525; arm totals 1,373,106 / 579,624 / 633,716; 1,280 aligned shared rows; zero skips); the 124-task gate frozen at seed 88,019 with zero prompt overlap against all six predecessor gates (88,013–88,018), the donor's corpora and streams, and both fresh training streams.

## Interpretation

None yet.

## Knowledgebase Update

- Program evidence updated: pending.
- Program backlog updated: this trial claims the queued interleaved-dose slot.
- Claim ledger updated: no.

## Artifacts

- `data/sft_hygiene_explore.jsonl`, `data/corpus_manifest.json`: inherited verified corpus.
- `data/stream_manifest.json`, `data/stream_token_receipt.json`: exposure receipts.
- `data/local_tasks_seed88019.jsonl`, `data/local_input_seed88019.jsonl`, `data/local_design_receipt.json`: frozen gate.
- `reports/preregistration.md`, `reports/design_review.md`: contract and authorization.
- `reports/artifact_manifest.yaml`: external parent and conditional model-artifact plan.
