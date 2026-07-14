# Adversarial Design Review

**Verdict:** pass after pre-registration revision; expensive work is authorized only
after this design checkpoint is committed and pushed.

## Primary threats and resolutions

1. **Dose interpolation can become post-result tuning.** The 160/240 doses are frozen
   from the known 80-row local failure and 400-row local-pass/broad-failure endpoints.
   Both doses run; neither may be chosen adaptively from partial local output.
2. **A 320-row arm was initially named but is not exact-match feasible without a new
   selection bias.** The source-token audit showed a proportional 320-row designed set
   is shorter than the 320 shortest replay rows. The arm was removed before data or
   model evaluation. A future length-aware arm must be result-separated.
3. **Matched rows and steps can still hide token compute.** Three disjoint 80-row
   designed blocks contain 33,613, 34,091, and 33,015 forward tokens. Their disjoint
   replay counterparts match those sums exactly. Every arm has 1,520 rows, 190
   updates, and exactly 1,405,510 forward tokens with zero skips.
4. **Dose may accidentally change skill coverage.** Every 80-row designed block covers
   all 13 truth-audited skills. The 160-row dose uses blocks A+B; the 240-row dose uses
   A+B+C. Tests assert exact position differences of 160, 80, and 240 rows.
5. **The stronger anchor could be misidentified.** All arms start from the parent
   replay-refresh adapter whose weights/config hashes are
   `c296c774...e3d36a` / `6b91df7d...9e4d6`. The training wrapper authenticates both.
6. **A forgiving local gate could leak weak arms into the benchmark.** Fresh seed
   88,005 retains the frozen absolute bars: accuracy ≥0.65, parse ≥0.90, at most
   two cap contacts, and no repeated feasible-route abstention. Each arm is judged
   independently; no arm passing means a deliberate nonzero stop.
7. **Multiple doses create winner's-curse risk.** Both doses are prospectively
   registered. Any locally eligible arm enters the same paired benchmark event and is
   reported separately. A passing pilot remains exploratory and requires independent
   confirmation with uncertainty and matched-compute sampling.
8. **Replay is an active treatment, not a neutral control.** A new replay-only arm
   uses the same parent, core, slots, tokens, updates, and seed. Designed candidates
   must beat it as well as base, `blend`, and inherited replay refresh.
9. **Backend or benchmark leakage could create a false universal result.** Local data
   are procedural and experiment-owned. The benchmark may be invoked only through the
   trusted aggregate gateway; all models use explicitly merged checkpoints and one
   `qwen_vllm` event. Private items, transcripts, and family sources remain unread.
10. **A pilot could be overstated.** Strict positive lift on all ten families is only
    a pilot promotion rule. A universal claim additionally requires fresh quick
    replication, medium@2,048, paired uncertainty, and a matched-compute sample-more
    comparison in a new confirmation experiment.

## Frozen artifact identities

- Source-token receipt SHA-256: `5821cf3c...602d5`.
- Dose manifest SHA-256: `7ea1f4ad...0b9a`.
- Token receipt SHA-256: `24bebb3c...f440`.
- Arm hashes: replay `1fd36d83...e1f45`, 160 rows `5159cf41...b40c8`,
  240 rows `c210d8cb...c05d5`.
- Construction seed: 77,109; training seed: 43; local seed: 88,005;
  conditional aggregate seed: 78,135.

No benchmark event or model training has run during this review.
