# Compute Review

Scoped to the six stage replays and the merge (~2.5–3 h GPU total).

- The stage plan is test-pinned against the byte-copied lineage manifest:
  per-stage dataset shas, fixed seeds 42/43/44/47/51/55 (deliberate
  reuse — the recipe IS the treatment's control), the three trainer
  variants at their recorded shas, full hyperparameters including stage
  3's targeted overrides; stage 1 fresh rank-32/alpha-64 with no
  warm-start path reachable; stages 2–6 warm-start strictly from the
  previous zero-root output; per-stage receipts record originals as
  contrast only.
- The rebuild stage requires this verdict committed at HEAD plus clean
  pushed green main; the merge uses the cell's own copied merger onto
  the raw pinned base; nothing in the rebuild path can touch the sealed
  benchmark seed.

**Verdict:** `PASS_REBUILD`.
