# Sweep-Rate Consolidation Experiment Log

## 2026-07-16 — Collection, analysis, errata, closure

- Six readings collected fail-closed from sha-pinned copied summaries
  and recomputed from per-family scores: 8/10, 10/10, 9/10, 8/10, 10/10,
  9/10 — zero strict losses in all sixty family comparisons; aggregate
  win 6/6.
- Sweep rate 2/6 = 0.333 (exact 95% CI [0.043, 0.777]; Beta posterior
  mean 0.375, 95% CrI [0.099, 0.710]). Menders blocks 4/4 misses (all
  0-margin draws); rites once; warren once (and warren WON +0.267 at
  78,155). Base draw note: base rites was 0.0 on all six seeds — the
  intake's example was wrong and is corrected here; chronicle is the
  family where base drew >0 on exactly the two passing seeds.
- Visible errata appended to the carrier documents (confirmation README
  + report, zero-root README, three synthesis paragraphs, and the brief/
  viz figures updated); per-seed facts unchanged everywhere.
- 30 tests green; --smoke and --full reproduce both artifacts
  byte-identically.
