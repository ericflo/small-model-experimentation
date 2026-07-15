# Retention-Screen Calibration Report

## Summary

The line's first instrument-calibration study is closed: five published composites across four fresh retention screens (20 authenticated eval events, zero training) measured the retention screen's delta-vs-parent noise at SD 4.27 — the ±5 single-screen band every prior gate used was ~1.2 SD wide. Outputs: `recommended_band` 9, `adjudication_protocol` `pooled_k3` (three fresh screens pooled before any retention adjudication; ±5 on the pooled mean is correctly sized at 2 × 4.27/√3 = 4.9), and all five historical single-screen "forgetting tax" readings (−9, −10, −10, −7, −5) inside measured noise around much smaller pooled deltas (−3.75, −2.25, −0.75, −0.75). The per-dose retention tax revises from 5–10 points to 1–4 points pooled; the rank-64 vehicle reads −0.75 vs rank-32's −3.75 descriptively (+3.0, within noise).

## Research Program Fit

The SCREEN_INSTABILITY verdict's funded successor; every future retention adjudication depends on it, and the paused rank-64 vehicle reading resumes descriptively under it.

## Method

See the preregistration.

## Results

`runs/local/calibration_readout.json`: delta_sd_pooled 4.27 (per-arm delta SDs 5.68 / 4.27 / 3.59 / 3.10), level SD 4.81 descriptive, band 9, protocol `pooled_k3`, five/five stability flags inside, vehicle descriptive +3.0 for rank 64. Full per-arm/per-screen table in the README.

## Controls

All arms published and weight-authenticated; screens overlap-receipted against all prior gates and each other; frozen run order; normalization unchanged.

## Oracle Versus Deployable Evidence

Executable truth grades outputs only; `benchmarks/` remains unread.

## Next Stage

Closed. Successors inherit the `pooled_k3` protocol: the vehicle question and any future dose gate adjudicate retention on the mean of three fresh screens with the ±5 band.

## Artifact Manifest

Four frozen screens in-repo; five composite pins external with committed receipts.
