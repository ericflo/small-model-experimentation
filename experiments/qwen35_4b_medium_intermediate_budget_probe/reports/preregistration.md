# Preregistration: Medium Intermediate Budget Probe

Frozen before any model event. Eval-only measurement intake: no training,
no promotion; one benchmark event that consumes one sealed seed and
closes. Minimal-delta successor of the reviewed tb8192 probe; every
post-review reading amendment carries over unchanged.

## Frozen identities

- Experiment: `qwen35_4b_medium_intermediate_budget_probe`.
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Arms — the same four explicit merged composites as the seed-78,150 and
  seed-78,152 events, tree-recomputed and bound at event time, frozen
  order: `base` (weights b654e033…, tree 26d8ee48…), `designed_fresh`
  (93433aa2…), `replay_repeat` (4c4f3561…), `hygiene_explore`
  (9eb653d7…).
- Event: tier `medium`, think budget **4,096** (the sole intentional
  delta from the tb8192 probe besides the seed), sealed fresh seed
  `78153`, trusted gateway only, hardened runner (review-verdict and
  code-pin checks at the seed boundary, write-ahead one-seed ledger,
  finiteness guards), identical benchmark source inventory across arms.

## Preregistered readings (no promotion bars; identical to the reviewed predecessor)

1. Budget movement — corrected premise carried over: at tb1024/78,150
   `menders` was 0 for ALL FOUR arms; `rites` was 0 only for base,
   replay_repeat, and hygiene_explore (designed_fresh scored 0.1). A
   movement boolean fires only for an (arm, family) pair whose pinned
   tb1024 value was 0 AND whose tb4096 value is > 0; designed_fresh's
   rites is excluded and reported descriptively. The goal-gate ceiling
   question (9 versus 10) is decided by menders for every arm, and by
   rites additionally for hygiene_explore and replay_repeat.
2. Budget contrast versus the pinned tb1024/78,150 summary — valid only
   after fail-closed benchmark-implementation-signature equality (runner
   a3beecd8…, inventory 218b8615…, count 56), both signatures surfaced;
   labeled `cross_seed_confound: true` (seed AND budget remain).
3. The goal gate recorded per treated arm (strict wins vs base across the
   ten public families; pass = 10).
4. Budget integrity: per-arm `within_budget` and wall seconds;
   `paired_comparison_valid: false` with reason if any arm exceeds.

## Preregistered stop outcome — the lever's last test

The gateway hard-fails an over-budget arm (`budget_gate_failed`). `base`
runs FIRST to minimize spend, and hygiene_explore (the slowest arm at
tb1024, 230 s) is as likely to bind — the stop applies whichever arm
trips: the event closes as `BUDGET_GATE_STOP`, failure receipts and any
completed arms' receipts preserved and reported descriptively, the seed
recorded as spent by the opened ledger record, no retry, no re-run at any
budget inside this directory. CONSEQUENCE (frozen): a second stop closes
the thinking-budget lever ENTIRELY for paired medium events — no further
budget probes at any setting without a new mechanism argument — and fixes
the statechain successor's 9/10 ceiling as the program's honest position.

## Mandatory checkpoint order

1. Model-free construction + adversarial design review — committed,
   pushed, green.
2. `benchmark` (requires `PASS_BENCHMARK_EVENT` in
   `reports/benchmark_design_review.md`, the design receipt committed at
   HEAD, clean pushed green main). No other stage exists.

## Interpretation limits

One seed, one tier, one budget; the contrast is seed-confounded by
construction and labeled as such; only within-event paired comparisons
are clean. Benchmark firewall unchanged.
