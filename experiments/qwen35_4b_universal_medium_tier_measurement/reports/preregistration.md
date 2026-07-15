# Preregistration: Universal-Line Medium-Tier Measurement

Frozen before any model event. Eval-only measurement intake: no training,
no promotion, no local gate; one benchmark event that consumes one sealed
seed and closes.

## Frozen identities

- Experiment: `qwen35_4b_universal_medium_tier_measurement`.
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Arms — four explicit merged composites, tree-hash recomputed and bound at
  event time, evaluated sequentially on the same seed in frozen order:
  1. `base` (reserialized) — tree `26d8ee48…b677`, weights `b654e033…16db`;
  2. `designed_fresh` — tree `93433aa2…0255`;
  3. `replay_repeat` — tree `4c4f3561…acd7`;
  4. `hygiene_explore` — tree `9eb653d7…4971`.
  Runtime LoRA is forbidden; every arm is an explicit merged composite.
- Event: tier `medium`, think budget 1,024, sealed fresh seed `78150`
  (grep-fresh audit in the design receipt), canonical gateway
  (`scripts/run_benchmark_aggregate.py`, sha pinned), identical benchmark
  source inventory across arms, one-seed ledger (the runner refuses a
  second event).

## Preregistered readings (no promotion bars)

1. Medium aggregate per arm, and the ordering compared against the frozen
   quick ordering from seed 78,144 (replay_repeat 0.5081 >
   designed_fresh 0.4644 > base 0.1085; hygiene_explore has no quick
   aggregate — recorded as null).
2. The goal gate RECORDED per treated arm: strict wins / ties / losses
   versus base across the ten public families; pass = ten strict wins.
   Frozen power statement: the forensics' historical medium pass rate is
   9/94 for arms that TRAINED on family data; these arms did not, so a
   pass is the program milestone (subject to the confirmation law:
   independent seeds + matched-compute sample-more before any claim), and
   a non-pass is expected and records the blocking families at the correct
   granularity.
3. Base sanity envelope: per family, whether base's medium score falls
   inside the historical base [min, max] from the forensics analysis
   (file sha pinned in the design receipt); an out-of-envelope base flags
   instrument drift and scopes every same-event comparison.
4. Blocking families per arm — the design constraints for the next dose.

## Mandatory checkpoint order

1. Model-free construction (design receipt, adversarial design review) —
   committed, pushed, green.
2. `benchmark` (requires `PASS_BENCHMARK_EVENT` in
   `reports/benchmark_design_review.md`, the design receipt committed at
   HEAD, and clean pushed green main). No other stage exists.

## Interpretation limits

One seed, one tier: readings are a map, not a claim. Cross-tier
comparisons use the frozen quick numbers from a different seed and carry
that caveat. The historical envelope comes from receipts run under earlier
budget configurations; it bounds sanity, not equivalence. Benchmark
firewall unchanged: `benchmarks/` is never read; only the gateway's
aggregate and public per-family scores are consumed.
