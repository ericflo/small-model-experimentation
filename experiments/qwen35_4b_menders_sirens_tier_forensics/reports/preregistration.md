# Preregistration: Menders/Sirens + Tier Forensics

Analysis-only. No model, no GPU, no seed is consumed, no benchmark content
is read; the inputs are the gateway OUTPUT receipts already committed under
`experiments/*/runs/`. Honest ordering note: this cell's questions and its
first-pass cleaning rules (drop non-score delta blocks by the unit-interval
test; dedupe identical blocks preferring named arms) were frozen in the
sweep/analyzer docstrings before results were interpreted; ONE cleaning
refinement — excluding `summary*.json` receipt files outright, because they
embed both copies of per-arm blocks and delta tables whose values can land
inside [0, 1] — was added after a first look exposed phantom rows. The
headline readings were unchanged by that refinement (goal-gate passes 9
medium / 1 quick under both cleanings), and both the raw table and the
cleaned analysis are committed so the sensitivity is checkable.

## Questions

1. Constant check: do "menders = 0" and "sirens = 0.500" hold for every arm
   at every seed, by tier and by think budget — and what are the exact
   counterexamples at the line's frozen quick/tb1024 instrument?
2. Base profile: per tier, the base arm's per-family score distribution,
   with floor (0) and ceiling (1) frequencies — a ceiling family makes a
   strict win against base impossible, a floor family needs only any score.
3. Goal-gate feasibility: within each (experiment, seed) event, pair every
   treated arm against the base arm and count strict wins across the ten
   public families; the goal gate is 10/10. Report the pass rate and the
   blocking families of 9/10 near-misses, by tier.

## Frozen mechanics

- Sweep: every `experiments/*/runs/**/*.json` (this cell excluded) parsed;
  every dict carrying all ten public family keys becomes a row with tier /
  think budget / seed recovered from the path, provenance sha256, and the
  sibling aggregate when present. Raw table committed as
  `runs/receipt_table.json`.
- Cleaning: as above (unit-interval filter, summary-file exclusion,
  identical-block dedupe). Arm class "base" = label contains "base".
- Outputs committed as `runs/constants_analysis.json`; the harness's smoke
  re-derives the analysis byte-identically from the committed table.

## Interpretation limits

The historical medium passes come from the earlier gym line, whose arms
TRAINED on menagerie-family data — they evidence instrument feasibility,
not transfer for the contamination-free universal line, which has never
been measured at medium. Historical receipts ran under their eras' think
budgets and backends; comparisons are paired within event only. Pass rates
are instrument evidence, not transferable probabilities.
