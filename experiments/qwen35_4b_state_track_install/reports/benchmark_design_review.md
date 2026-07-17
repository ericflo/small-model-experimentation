# Benchmark Event Adversarial Review

The conditional stage unlocked by the committed local promotion:
state_track retained within all pooled_k3 bands vs the count_walk parent
at seeds 88063-88065. Backed by the three-lens adversarial workflow
(zero MAJORs at design; the 1e-12 aggregate tie guard and the standalone
fixes carried from lifecycle 29).

- All three normalized-pin slots filled from the committed merge receipt
  and verified against the exact runtime authentication: merge-receipt
  pin = sha256 of the committed runs/merges/state_track.json (089f280e…,
  the FILE sha the runner compares — the lifecycle-29 lesson applied),
  tree pin = the receipt's output_tree_sha256 (45fd2925…), weights pin =
  the receipt's model.safetensors sha (b4bafbb7…). check_design --check
  confirms the three-slot normalized pin holds.
- The parent (count_walk) and base pins are frozen design-time
  constants; the parent is authenticated via the in-cell provenance
  copy (sibling original a verification aid only). Gateway sha pinned
  and verified against scripts/run_benchmark_aggregate.py.
- Event: medium, tb 1,024, sealed fresh seed 78,169, three arms in
  frozen order (base, count_walk, state_track), the hardened
  receipt-pinned write-ahead ledger. The frozen two-state
  INSTALLED_TRANSFER / BOUNDED consequence applies (candidate aggregate
  strictly > parent AND > base with the tie guard, every family within
  one episode of parent). Honest prior P(INSTALLED_TRANSFER) ≈ 0.30-0.40;
  BOUNDED is a real finding (extends install≠convert to this skill).
  Goal gate vs base recorded descriptively.

**Verdict:** `PASS_BENCHMARK_EVENT`.
