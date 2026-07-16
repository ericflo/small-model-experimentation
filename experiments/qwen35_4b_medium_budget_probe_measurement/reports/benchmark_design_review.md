# Benchmark Event Adversarial Review

Eval-only budget probe adapting the twice-reviewed hardened runner; the
delta review (two lenses, adversarial verification) confirmed two MAJORs,
both fixed and re-verified pre-freeze:

1. The movement premise was falsified by the cell's own pinned contrast
   source — designed_fresh already scored rites 0.1 at tb1024, so the
   absolute >0 boolean could fire on a status-quo repeat. Movement is now
   scoped: a boolean fires only for an (arm, family) pair at exactly 0 in
   the pinned tb1024 event whose tb8192 value is positive; already-nonzero
   pairs are hard-excluded and reported descriptively; a status-quo
   repeat fires nothing (regression-tested).
2. The cross-budget contrast never verified benchmark-implementation
   equality across events. The readings now fail closed unless the four
   new receipts' shared (runner sha, inventory sha, file count) signature
   matches the reference summary's benchmark_implementation block —
   verified against the real pinned values (runner a3beecd8…, inventory
   218b8615…, count 56) — with both signatures surfaced and the confound
   note corrected (seed and budget remain; implementation verified).

Inherited hardening confirmed live: the verdict gate and the receipt's
code-pin `--check` sit at the seed boundary (direct invocation exits
before any gateway call with `runs/` untouched); write-ahead
opened/closed ledger; per-arm tree recompute; finiteness guards; the
preregistered `BUDGET_GATE_STOP` outcome covers the gateway's hard
budget-gate failure with base ordered first to minimize spend. 75 tests
green; smoke green; receipt `569ab423…` `--check` byte-identical twice;
seed 78,152 audit fresh.

**Verdict:** `PASS_BENCHMARK_EVENT`.
