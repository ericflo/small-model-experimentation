# Rank-Capacity Vehicle Cell

The vehicle study's sharpest single variable: a FRESH rank-64 adapter trained on the clean parent composite with the same twice-verified corpus and exposure, judged on a fresh screen beside the published rank-32 arm (known −9 retention) and the parent — asking whether capacity removes the intrinsic retention tax while preserving the install.

**Status:** in-progress · since 2026-07-15 · model-free construction under way; no model event has run

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: the priced intrinsic-tax law (diversity and interleaving both refuted; screen fortune resolved); the interference arc's independent capacity implication; hygiene seven-for-seven as the install probe.

## Question

Does doubling adapter capacity (rank 64, fresh adapter on the composite) absorb the dose without evicting retained skills (CAPACITY_SUPPORTED), fail to (CAPACITY_REFUTED), or does the known −9 fail to reproduce (SCREEN_INSTABILITY)?

## Hypothesis

The tax is subspace competition: the rank-32 adapter carries the whole lineage, and a new dose must evict something. A fresh rank-64 adapter on the frozen composite separates the dose's parameters from the lineage's entirely — the cleanest capacity test available.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Base for training: the `designed_fresh` merged composite (weights `0a3b89cd...979`) via the per-experiment trainer's new `--model-path` argument (encode_row byte-unchanged); FRESH rank-64/alpha-128 adapter, no warm start; otherwise identical geometry (1,520 rows, 190 updates, LR 1e-5, seed 58; inherited corpus `e7a95d73...79e`; slot seed 55,124).
- Merge: the rank-64 adapter applied onto the composite weights (scale 2.0, fingerprint and module checks unchanged).
- Gate (seed 88,021): 40-task v1-kind axis holdout + 104-task retention; three weight-authenticated arms (`axis160_r64`, published `axis160_direct`, `clean_parent`); ordered verdict partition — SCREEN_INSTABILITY (the −9 fails to reproduce at ≥ −5) takes precedence, then CAPACITY_SUPPORTED (r64 ≥ −5 while r32 ≤ −6), else CAPACITY_REFUTED; plus an `install_preserved` flag (r64 axis total ≥ r32's on this screen).
- No benchmark stage and no aggregate seed.

## Run

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_rank_capacity_vehicle_cell/scripts/run.py --smoke
```

Checkpointed stages:

```bash
.venv/bin/python -B experiments/qwen35_4b_rank_capacity_vehicle_cell/scripts/run.py --stage train-candidate
.venv/bin/python -B experiments/qwen35_4b_rank_capacity_vehicle_cell/scripts/run.py --stage merge-candidate
.venv/bin/python -B experiments/qwen35_4b_rank_capacity_vehicle_cell/scripts/run.py --stage local
```

## Results

No model event has run.

## Interpretation

None yet.

## Knowledgebase Update

- Program evidence updated: pending the verdict.
- Program backlog updated: this cell is the vehicle study's first variable.
- Claim ledger updated: no.

## Artifacts

- `data/sft_axis160.jsonl`, `data/corpus_manifest.json`: inherited twice-verified corpus.
- `data/stream_manifest.json`, `data/stream_token_receipt.json`: exposure receipts.
- `data/local_tasks_seed88021.jsonl`, `data/local_input_seed88021.jsonl`, `data/local_design_receipt.json`: frozen gate.
- `reports/preregistration.md`, `reports/design_review.md`: contract and authorization.
- `reports/artifact_manifest.yaml`: external composite pins.
