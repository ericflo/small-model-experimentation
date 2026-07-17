# Benchmark Event Adversarial Review

The conditional stage unlocked by the committed local promotion:
replay_compound retained within all pooled_k3 bands vs the count_walk
parent at seeds 88060-88062. Backed by the three-lens adversarial
workflow (both MAJORs fixed and re-verified at e61915e2); the aggregate
comparison now carries the 1e-12 tie guard, so a true rational tie
resolves as BOUNDED.

- All three normalized-pin slots filled from the committed merge
  receipt and verified against the exact runtime authentication:
  merge-receipt pin = sha256 of the committed
  runs/merges/replay_compound.json (4170b082…, the FILE sha the runner
  compares — an orchestration slip pinning the internal
  merge_receipt_sha256 field was caught and corrected before this
  review), tree pin = the receipt's output_tree_sha256 (22a045e0…),
  weights pin = the receipt's model.safetensors sha (65d12d18…).
  check_design --check confirms the three-slot normalized pin holds.
- The parent (count_walk) and base pins are frozen design-time
  constants; the parent is authenticated via the in-cell provenance
  copy (sibling original a verification aid only). Gateway sha pinned
  and verified against scripts/run_benchmark_aggregate.py.
- Event: medium, tb 1,024, sealed fresh seed 78,168, three arms in
  frozen order (base, count_walk, replay_compound), the hardened
  receipt-pinned write-ahead ledger. The frozen two-state
  COMPOUNDED/BOUNDED consequence applies (candidate aggregate strictly
  > parent AND > base with the tie guard, every family within one
  episode of parent); honest priors favor BOUNDED as the
  believed-likelier finding. Goal gate vs base recorded descriptively.

**Verdict:** `PASS_BENCHMARK_EVENT`.
