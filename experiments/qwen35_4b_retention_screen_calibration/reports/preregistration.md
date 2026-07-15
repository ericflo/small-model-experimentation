# Preregistration: Retention-Screen Calibration Study

Frozen before any model event. Eval-only: no training, no promotion, no
benchmark seed, no claim. The outputs govern future adjudications.

## Frozen identities

- Experiment: `qwen35_4b_retention_screen_calibration`.
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Arms — five published explicit composites, weight-recomputed at event time
  against their committed merge receipts: `clean_parent` (designed_fresh),
  `replay_clean`, `hygiene_explore_direct`, `axis160_direct`, `axis160_r64`.
- Screens: seeds `88022 / 88023 / 88024 / 88025`, each 104 retention rows
  (8 per original skill), oracle-free, overlap-receipted against all prior
  gates (88,013–88,021), each other, and every inherited corpus.
- Run order (frozen): screen-major, seeds ascending, arms alphabetical within
  a screen — 20 authenticated engine events, standard boundary re-auth,
  answer normalization unchanged.

## Preregistered outputs

From the consolidated receipt's per-arm/per-screen retention-correct table:

1. `delta_sd_pooled` — the governing estimand: the pooled across-screen
   standard deviation of the per-screen retention-correct delta versus
   `clean_parent`, pooled over the four non-parent arms (sample SD, ddof=1,
   df=3 per arm). Every retention band this program adjudicates is a
   same-screen delta versus a parent or control, so the band is calibrated
   on the delta noise process: common screen-difficulty variance cancels in
   same-screen deltas while independent per-arm noise widens them by ~√2,
   making the level SD the wrong estimand in both directions (adversarial
   design review finding, corrected before freeze — the original draft
   derived the band from the level SD).
2. `screen_sd_pooled`: the pooled within-arm across-screen standard deviation
   of retention-correct levels over all five arms — reported descriptively;
   it governs nothing.
3. `recommended_band`: ⌈2 × delta_sd_pooled⌉, minimum 5.
4. `adjudication_protocol`: `single_screen` if delta_sd_pooled ≤ 2;
   `pooled_k2` if ≤ 3.5; else `pooled_k3` — the number of independent fresh
   screens future cells must pool before adjudicating a retention band.
5. Historical stability flags: whether the −9 (axis160_direct at 88,020), the
   −10s (hygiene_explore_direct at 88,018/88,020), the −5 (replay_clean at
   88,020), and the −7 (axis160_r64 at 88,021) fall inside their arms'
   intervals from this study — pooled delta versus clean_parent ± 2 × the
   across-screen sample SD (ddof=1) of that arm's per-screen delta series.
6. The paused vehicle reading, resumed descriptively: axis160_r64's pooled
   delta versus axis160_direct's, with measured-noise intervals — reported,
   not gated.

## Mandatory checkpoint order

1. Model-free construction + design review — committed, pushed, green.
2. `local` (requires `PASS_LOCAL_EVENT` in reports/local_design_review.md).
   No other stage exists.

## Interpretation limits

Four screens estimate SD with wide uncertainty themselves (df=3 per arm;
pooling across the non-parent arms mitigates); the four delta series share
the parent arm's screen draws, so they are positively correlated and the
pooled delta SD's nominal df of 12 overstates the information — the outputs
are instrument calibration, not inference; benchmark firewall unchanged.
