# Retention-Screen Calibration Study

Measure the measuring stick: five published composites re-run across four fresh 104-task retention screens (20 authenticated eval events, zero training) to size the screen's seed-to-seed variance directly, derive the band that separates real retention effects from draws, and freeze the adjudication protocol every future dose and vehicle cell will use.

**Status:** finished · 2026-07-15 · verdict CALIBRATION_READ_COMPLETE — the retention screen's delta-vs-parent noise is SD 4.27, so the ±5 single-screen band was ~1.2 SD wide; all five historical tax readings sit inside measured noise (pooled taxes only 1–4 points); frozen protocol pooled_k3: three fresh screens, ±5 band on their mean

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

All 20 runs completed and weight-authenticated (receipt `runs/local/calibration.json`, readout `runs/local/calibration_readout.json`):

| arm | correct by screen (88022/23/24/25) | delta vs parent by screen | pooled delta | delta SD |
|---|---|---|---|---|
| clean_parent | 69 / 67 / 69 / 64 | — | — | — |
| axis160_direct | 62 / 57 / 69 / 66 | −7 / −10 / 0 / +2 | **−3.75** | 5.68 |
| axis160_r64 | 65 / 71 / 69 / 61 | −4 / +4 / 0 / −3 | **−0.75** | 3.59 |
| hygiene_explore_direct | 72 / 66 / 65 / 57 | +3 / −1 / −4 / −7 | **−2.25** | 4.27 |
| replay_clean | 70 / 69 / 68 / 59 | +1 / +2 / −1 / −5 | **−0.75** | 3.10 |

- Governing estimand `delta_sd_pooled` = **4.27** → `recommended_band` = **9**, `adjudication_protocol` = **`pooled_k3`** (every future retention adjudication must pool three fresh screens). Level SD 4.81 (descriptive).
- Stability flags: **all five** historical single-screen readings (−9 axis160_direct@88020, −7 axis160_r64@88021, −10/−10 hygiene_explore_direct@88018/88020, −5 replay_clean@88020) fall **inside** their arms' pooled-delta ± 2·SD intervals.
- Vehicle (descriptive, not gated): rank-64 pooled delta −0.75 vs rank-32's −3.75 (difference +3.0 favoring rank 64, within noise).

## Interpretation

- The ±5 single-screen retention band every prior gate used was ~1.2 SD wide — a true-zero-cost arm fails it roughly one screen in eight, and the program adjudicated four arms per event. The rank cell's SCREEN_INSTABILITY guard is vindicated and quantified.
- The "intrinsic retention tax ~5–10 points per dose" reading must be revised: pooled over four screens the taxes are 0.75–3.75 points; the historical 5–10-point readings were single-screen draws from a ±4.3-SD process around those small means. A tax likely exists (all four pooled deltas are negative) but is several times smaller than the single-screen readings suggested.
- Coherence check: ±5 is almost exactly 2 × (4.27/√3) = 4.9 — the historical band size is right if and only if it is applied to the MEAN of three pooled fresh screens, which is precisely the frozen `pooled_k3` protocol.
- Screen 88025 ran hard for every arm (parent 64 vs 67–69 elsewhere) — a real common-difficulty component that same-screen deltas cancel; the estimand correction from the adversarial review (delta SD, not level SD) is what kept this event's outputs from being inflated by it.

## Knowledgebase Update

- Program evidence updated: calibration outputs + the revised tax reading recorded.
- Program backlog updated: instability successor closed; vehicle resumption and any future dose gate now require the `pooled_k3` protocol.
- Claim ledger updated: no new claim; the tax law's evidence note revised in synthesis.

## Artifacts

- `data/local_tasks_seed8802{2,3,4,5}.jsonl` + inputs + `data/local_design_receipt.json`: four frozen screens.
- `reports/preregistration.md`, `reports/design_review.md`: contract and authorization.
- `reports/artifact_manifest.yaml`: the five composite pins.
