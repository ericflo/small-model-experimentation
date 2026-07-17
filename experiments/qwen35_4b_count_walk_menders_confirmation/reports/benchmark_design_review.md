# Benchmark Event Adversarial Review

The four-seed eval-only confirmation, authorized after a two-lens
adversarial review (recorded in experiment_log.md) found 3 MAJORs and
all were fixed and re-frozen pre-event at 3fbbf3d6 (no seed consumed):

- Episode conversion is floor semantics (banker's rounding on the k/60
  partial-credit lattice could manufacture phantom episodes and flip
  AMBIGUOUS→REPLICATED; 61-point lattice sweep unit-tested); hits count
  only events with ≥1 FULL episode, so the frozen rule coincides with
  the preregistered pricing (α 0.0450 at p=0.10, 0.0475 exact at 3/29;
  REPLICATED power 0.4717/0.6289/0.8230 at q=0.4/0.5/0.65, all
  --check-enforced).
- The reviewer live-verified every frozen constant against reality
  (full 36GB tree+weights recompute of all four composites
  authenticated on disk; gateway sha, prior-summary sha, merge-receipt
  shas, provenance copies byte-identical) and ran mutation probes
  (tampered pins, tampered provenance, ledger re-open, spent budget,
  off-list seed) — all refused fail-closed. Crash walks reconcile
  byte-equal; pre-consumption implementation-signature check added.
- The full lineage package is in-cell per the eval-only doctrine
  clause (rebuild_lineage.py --verify-inputs green over 27 pinned
  copies).
- Event: four sealed fresh medium tb1024 seeds 78164/78165/78166/78167,
  four arms each in frozen order (base, zero_root_parent, replay_ctl7,
  count_walk), k-seed write-ahead ledger, one-time consumption. The
  frozen three-state replication rule and both negative/ambiguous
  consequences are preregistered two-directionally; 78163 is prior
  evidence and never pooled; all per-family/aggregate/goal-gate
  readings are descriptive, never gating.

**Verdict:** `PASS_BENCHMARK_EVENT`.
