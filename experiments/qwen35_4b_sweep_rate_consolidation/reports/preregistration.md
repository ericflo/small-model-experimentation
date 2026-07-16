# Preregistration: Sweep-Rate Consolidation

Analysis-only; no model, no GPU, no seed; inputs are six committed
gateway summaries, byte-copied into `data/source_summaries/` and
sha-pinned (the cell verifies its local copies against the pins and,
when present, against the originals). Honest ordering note: the six
readings' values were already informally known from the source cells;
this cell's contribution is the fail-closed collection, the exact
uncertainty, the blocker table, base's draw distribution, and the
ERRATUM machinery — all recomputed from per-family scores with the
forensics-identical FAMILIES tuple and cross-checked against any
recorded goal-gate block.

## Frozen outputs

1. The six readings (seeds 78,150/78,154/78,155/78,156/78,157/78,159):
   strict wins/ties/losses versus base per seed, with provenance shas.
2. Sweep rate over all six with the exact Clopper–Pearson 95% CI and a
   Beta(1,1)-posterior summary.
3. Blocker frequencies across misses and the zero-strict-losses fact.
4. Base's per-family draw distribution across the six seeds (computed,
   not assumed).
5. The erratum block: the informal window figures (2/4 "~50%"; the
   "fifth data point" 2/5 framing), the corrected all-events figure, and
   the pinned list of carrier documents receiving visible errata.

## Interpretation limits

Six readings bound the rate loosely by construction (the CI is wide and
reported); the readings share one composite and one tier/budget; nothing
here changes any per-seed fact — only the aggregation is corrected.
