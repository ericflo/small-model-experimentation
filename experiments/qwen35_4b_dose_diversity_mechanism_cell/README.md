# Dose-Diversity Mechanism Cell

The single missing measurement that adjudicates why designed doses cost retention: the verified 160-row/4-kind corpus dosed DIRECTLY from the clean parent (no replay round), judged on a fresh screen alongside the re-measured 80-row/2-kind dose (known −10), the replay round, and the parent — with a preregistered three-way verdict.

**Status:** in-progress · since 2026-07-15 · model-free construction under way; no model event has run

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: the interleaving refutation and its escalation rule; the sole retention-safe dose event (160-row corpus) versus two ~−10 events (80-row corpus); hygiene six-for-six as the install probe.

## Question

Does corpus size/diversity at matched per-kind dose protect retention (SUPPORTED), is the retention cost intrinsic to dosing this vehicle (REFUTED_INTRINSIC), or does the known −10 fail to reproduce on a fresh screen (SCREEN_FORTUNE_SUSPECT)?

## Hypothesis

Diversity: the 160/4-kind block dilutes per-kind gradient pressure on the shared representation that retained skills live on. Each alternative outcome selects a different, preregistered successor.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: `designed_fresh` composite (tree `93433aa2...255`); warm start from its adapter (`36f41095...442`).
- New arm: `axis160_direct` — the twice-verified 160-row v1 axis corpus (sha `e7a95d73...79e`), inherited byte-identically, in the standard exact-exposure stream (slot seed 55,123; training seed 57; 190 updates).
- Gate (seed 88,020): 40-row v1-kind axis holdout + 104-row retention screen; four arms — the new merge plus three published composites (`hygiene_explore_direct` with its known −10, `replay_clean`, `clean_parent`); normalization unchanged; NO promotion — the receipt carries the per-arm table and the three-way `diversity_mechanism` verdict (bands: SUPPORTED if axis160 retention ≥ parent−5 while hygiene_explore ≤ parent−6; REFUTED_INTRINSIC if axis160 ≤ parent−6; SCREEN_FORTUNE_SUSPECT if the −10 fails to reproduce at ≥ parent−5).
- No benchmark stage and no aggregate seed: a mechanism cell mints no claims.

## Run

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_dose_diversity_mechanism_cell/scripts/run.py --smoke
```

Checkpointed stages:

```bash
.venv/bin/python -B experiments/qwen35_4b_dose_diversity_mechanism_cell/scripts/run.py --stage train-candidate
.venv/bin/python -B experiments/qwen35_4b_dose_diversity_mechanism_cell/scripts/run.py --stage merge-candidate
.venv/bin/python -B experiments/qwen35_4b_dose_diversity_mechanism_cell/scripts/run.py --stage local
```

## Results

No model event has run.

## Interpretation

None yet.

## Knowledgebase Update

- Program evidence updated: pending the verdict.
- Program backlog updated: this cell is the escalation rule's funded successor.
- Claim ledger updated: no.

## Artifacts

- `data/sft_axis160.jsonl`, `data/corpus_manifest.json`: inherited twice-verified corpus.
- `data/stream_manifest.json`, `data/stream_token_receipt.json`: exposure receipts.
- `data/local_tasks_seed88020.jsonl`, `data/local_input_seed88020.jsonl`, `data/local_design_receipt.json`: frozen gate.
- `reports/preregistration.md`, `reports/design_review.md`: contract and authorization.
- `reports/artifact_manifest.yaml`: external composite pins.
