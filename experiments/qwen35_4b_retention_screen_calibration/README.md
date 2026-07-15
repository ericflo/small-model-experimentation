# Retention-Screen Calibration Study

Measure the measuring stick: five published composites re-run across four fresh 104-task retention screens (20 authenticated eval events, zero training) to size the screen's seed-to-seed variance directly, derive the band that separates real retention effects from draws, and freeze the adjudication protocol every future dose and vehicle cell will use.

**Status:** in-progress · since 2026-07-15 · model-free construction under way; no model event has run

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: the SCREEN_INSTABILITY verdict (the known −9 re-measured at −5); pooled same-composite scatter of ±3–4 points across four gates; the paused rank-64 vehicle question awaiting calibrated bands.

## Question

What is the retention screen's true per-arm seed variance, what band does it imply, and which historical single-screen readings survive pooled re-measurement?

## Hypothesis

Screen SD is on the order of 2–4 points, implying either wider bands or pooled multi-screen adjudication; the dose-tax direction survives pooling while individual band-edge calls do not.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Arms (published, weight-authenticated): `clean_parent`, `replay_clean`, `hygiene_explore_direct`, `axis160_direct`, `axis160_r64`.
- Screens: four frozen retention-only instruments (seeds 88,022–88,025; 104 rows each from the original 13-skill generator), oracle-free inputs, overlap-receipted against all prior gates and each other.
- Event: 20 sequential authenticated engine runs in frozen screen-major order; consolidated receipt with per-arm/per-screen table, the governing pooled delta-vs-parent SD (level SD reported descriptively), recommended band (⌈2·delta SD⌉, min 5), adjudication protocol tier, and historical stability flags.
- No training, merging, promotion, or benchmark stage.

## Run

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_retention_screen_calibration/scripts/run.py --smoke
```

Checkpointed stage:

```bash
.venv/bin/python -B experiments/qwen35_4b_retention_screen_calibration/scripts/run.py --stage local
```

## Results

No model event has run.

## Interpretation

None yet.

## Knowledgebase Update

- Program evidence updated: pending calibration.
- Program backlog updated: this study is the instability verdict's funded successor.
- Claim ledger updated: no.

## Artifacts

- `data/local_tasks_seed8802{2,3,4,5}.jsonl` + inputs + `data/local_design_receipt.json`: four frozen screens.
- `reports/preregistration.md`, `reports/design_review.md`: contract and authorization.
- `reports/artifact_manifest.yaml`: the five composite pins.
