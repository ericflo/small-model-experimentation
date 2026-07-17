# Qwen35 4b Count Walk Menders Confirmation Experiment Log

## 2026-07-17 — design freeze (lifecycle 28, eval-only)

- Scaffolded as the mandatory confirmation cell for lifecycle 27's
  MECHANISM_ANSWER (count_walk menders 0.1 vs base / zero_root_parent /
  replay_ctl7 all 0.0 at sealed seed 78,163).
- Seed-freshness audit: 78,164 / 78,165 / 78,166 / 78,167 verified grep-fresh
  in seed contexts across the repo (every raw numeric hit is a float/sha
  substring in unrelated data files); benchmark seeds previously spent through
  78,163; no substitution required.
- Frozen the integer-exact two-directional replication rule (REPLICATED /
  NOT_REPLICATED / AMBIGUOUS, no fourth state) with all three claims worded in
  the preregistration, and the exact power arithmetic (false-REPLICATED 0.0450
  under the p=0.10 noise model, sensitivity 0.0947; power of hits_c >= 2:
  0.5248 / 0.6875 / 0.8735 at q = 0.4 / 0.5 / 0.65; full REPLICATED power
  0.4717 / 0.6289 / 0.8230), recomputed fail-closed by
  `scripts/power_analysis.py --check`.
- Cloned and adapted the hardened runner machinery: fail-closed tree+weights
  authentication of the four pre-existing composites (constants baked at design
  time, no TODO pins), lifecycle-27 merge receipt + lifecycle-22 zero-root
  provenance authentication, gateway sha pin, k-seed write-ahead opened/closed
  ledger with byte-equal crash reconciliation, implementation-signature
  equality against the pinned prior event, ledger-anchored terminal readout.
- Copied the four committed provenance documents byte-identically into
  `data/provenance/` as verification aids; composite reproduction remains
  lifecycle 27's / lifecycle 22's own standalone rebuild path (this cell
  produces no model); the measurement gateway stays shared per
  docs/quality_gates.md.
- Unit tests: replication-rule truth table (including the E_c tie branch),
  ledger open/close/reconcile/double-consume refusals, arm-authentication
  failure paths, frozen constants, readout schema, finiteness guards, power
  arithmetic. Smoke green; no GPU stage run; no seed consumed.
- Next checkpoint: adversarial benchmark design review
  (`reports/benchmark_design_review.md` with the literal
  PASS_BENCHMARK_EVENT verdict) before `--stage benchmark` can consume any
  seed.
