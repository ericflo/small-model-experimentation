# Dose-Diversity Mechanism Cell

The single missing measurement that adjudicates why designed doses cost retention: the verified 160-row/4-kind corpus dosed DIRECTLY from the clean parent (no replay round), judged on a fresh screen alongside the re-measured 80-row/2-kind dose (known −10), the replay round, and the parent — with a preregistered three-way verdict.

**Status:** finished · 2026-07-15 · verdict REFUTED_INTRINSIC — the retention cost of designed doses is intrinsic to this vehicle (diverse dose −9; known −10 reproduced; replay itself −5); the sole retention-safe precedent was screen fortune; hygiene reached seven-for-seven

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

The single arm trained cleanly (0.5068 train loss, 0 skips) and merged. The four-arm gate at seed 88,020 (fresh screen, normalized grading): retention correct of 104 — clean_parent 70, replay_clean 65 (−5), axis160_direct 61 (−9), hygiene_explore_direct 60 (−10, reproducing its known cost exactly). Preregistered verdict: **REFUTED_INTRINSIC** — the diverse dose broke the band too. Axis holdout of 40: axis160_direct 26 (best; hygiene 10/10 — the seventh consecutive hygiene win, now perfect; caps 5, best), hygiene_explore_direct 24, clean_parent 24, replay_clean 23.

## Interpretation

The mechanism question is answered: at this vehicle (rank-32 LoRA continued in place, 190 updates, LR 1e-5), designed doses cost roughly five to ten retention points intrinsically — corpus diversity does not protect it, replay interleaving does not protect it (prior refutation), and even a pure replay round costs about five on a fresh screen. The single retention-byte-equal precedent was screen fortune, exactly as the SCREEN_FORTUNE alternative anticipated for the OTHER arm. Meanwhile the installs themselves are unambiguous: hygiene is now seven-for-seven across every parent, dose size, and recipe, and the diverse dose posted the best axis total and termination in this event. The program-level law: install-versus-retention is a real, priced trade at this vehicle. Successors must either change the vehicle (rank, loss weighting, update count) or preregister gates that price the trade rather than demand its absence.

## Terminal Disposition

No later event is authorized here. No benchmark seed existed. All four composites and the verdict receipt are preserved. Per the preregistered branch, the funded successor is a dose-vehicle study (rank / loss weights / update count as single variables against this same gate design), with its own intake.

## Knowledgebase Update

- Program evidence updated: the intrinsic-cost verdict, the screen-fortune resolution, and hygiene's seven-for-seven recorded.
- Program backlog updated: the vehicle study is the funded successor; the recipe search stays closed.
- Claim ledger updated: no.

## Artifacts

- `data/sft_axis160.jsonl`, `data/corpus_manifest.json`: inherited twice-verified corpus.
- `data/stream_manifest.json`, `data/stream_token_receipt.json`: exposure receipts.
- `data/local_tasks_seed88020.jsonl`, `data/local_input_seed88020.jsonl`, `data/local_design_receipt.json`: frozen gate.
- `reports/preregistration.md`, `reports/design_review.md`: contract and authorization.
- `reports/artifact_manifest.yaml`: external composite pins.
