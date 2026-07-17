# Benchmark Event Adversarial Review

The six-seed eval-only confirmation, authorized after a two-lens
adversarial review (recorded in experiment_log.md) of the frozen,
model-free design (no seed consumed — the ledger does not exist and no
gateway call has ever run from this cell):

- The frozen rule is the PAIRED aggregate delta
  `d_i = state_track_aggregate − count_walk_aggregate` on the same seed
  (common seed-variance cancels — the whole point of pairing, and the
  fix for the specific failure mode this cell exists to check: the
  parent's aggregate swings 0.30–0.36 seed-to-seed). CONFIRMED iff
  `mean_d > 0` AND `wins >= 4`; NOT_CONFIRMED iff `mean_d <= 0` (the mean
  clause dominates the `wins >= 4` edge); AMBIGUOUS otherwise. All three
  branches, the tie-guard branches, and the total-partition sweep are
  unit-tested; the 1e-12 aggregate tie guard (carried from lifecycle 30)
  separates ulp noise from real per-event differences (≥ ~1.7e-3) by nine
  orders of magnitude.

- Power is honest and quoted two-directionally. The reviewer required and
  the preregistration states PLAINLY that the paired majority rule is a
  LIBERAL directional check: under the pure null the false-CONFIRMED rate
  is 0.311 (deterministic-convolution joint, bounded by the exact
  P(wins>=4)=0.34375 and the independence product 0.17188; scale-free
  under mu=0). The design's high-value outcome is the decisive
  NOT_CONFIRMED (mean ≤ 0), which retires the single-seed headline as seed
  noise. Under the observed +0.0256 lift the test has 90–98% power across
  sigma_d ∈ {0.02, 0.025, 0.03} (rho ∈ {0.78, 0.65, 0.50}); the sigma_d
  arithmetic (`sigma_d = sigma_arm·sqrt(2(1−rho))`, sigma_arm=0.03) is
  shown and every number is `power_analysis.py --check`-enforced.

- Both composites pre-exist this cell and every pin is a design-time
  constant (no TODO-PIN slot). The reviewer confirmed the pins against
  reality: the full tree+weights recompute of both composites
  authenticates on disk; the gateway sha, the prior-summary sha, and both
  merge-receipt file shas match; the in-cell provenance copies are
  byte-identical to their committed sibling originals (siblings are
  verification aids — the in-cell pin is authoritative). Mutation probes
  (tampered tree/weights pins, tampered provenance sha, ledger re-open,
  spent budget, off-list seed 78169, over-budget arm) all refuse
  fail-closed. Crash walks reconcile byte-equal; the pre-consumption
  implementation-signature check anchors all twelve receipts to the
  seed-78169 instrument.

- The full stage 1-9 lineage package is in-cell per the eval-only
  standalone doctrine (`rebuild_lineage.py --verify-inputs` green over
  25 pinned copies + 7 provenance receipts; the stage-9 `extended_by`
  identity stays the source cell `qwen35_4b_state_track_install`).

- Event: six sealed fresh medium tb1024 seeds
  78170/78171/78172/78173/78174/78175, two arms each in frozen order
  (count_walk, state_track), k-seed write-ahead ledger, one-time
  consumption. The frozen three-state paired rule and both
  negative/ambiguous consequences are preregistered two-directionally;
  78169 is prior evidence and never pooled; all per-family / aggregate /
  goal-gate readings are descriptive, never gating.

**Verdict:** `PASS_BENCHMARK_EVENT`.
