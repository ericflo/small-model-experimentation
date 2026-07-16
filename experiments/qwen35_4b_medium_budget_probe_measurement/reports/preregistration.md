# Preregistration: Medium Budget-Probe Measurement

Frozen before any model event. Eval-only measurement intake: no training,
no promotion; one benchmark event that consumes one sealed seed and
closes.

## Frozen identities

- Experiment: `qwen35_4b_medium_budget_probe_measurement`.
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Arms — the SAME four explicit merged composites as the seed-78,150
  event, tree-recomputed and bound at event time, frozen order: `base`
  (weights b654e033…), `designed_fresh` (93433aa2…), `replay_repeat`
  (4c4f3561…), `hygiene_explore` (9eb653d7…).
- Event: tier `medium`, think budget **8,192** (the probe lever; the only
  intentional difference from the reference event), sealed fresh seed
  `78152`, trusted gateway only, hardened runner (review-verdict and
  code-pin checks at the seed boundary, write-ahead one-seed ledger,
  finiteness guards), identical benchmark source inventory across arms.

## Preregistered readings (no promotion bars)

1. Budget movement — corrected premise (review amendment, pre-freeze):
   at tb1024/78,150 `menders` was 0 for ALL FOUR arms, while `rites` was 0
   only for base, replay_repeat, and hygiene_explore (designed_fresh
   scored 0.1 there, already a strict win). A movement boolean therefore
   fires only for an (arm, family) pair whose pinned tb1024 value was 0
   AND whose tb8192 value is > 0; designed_fresh's rites is excluded from
   the booleans and reported descriptively. The headline cannot fire on a
   status-quo repeat. The goal-gate ceiling question (9 versus 10) is
   decided by menders for every arm, and by rites additionally for
   hygiene_explore and replay_repeat.
2. Budget contrast: per arm per family, delta versus the committed
   tb1024 event at seed 78,150 (summary sha-pinned; fail closed on
   change), valid only after the two events' benchmark implementations
   are verified equal — the new receipts' shared (runner sha, source
   inventory sha, file count) signature must match the reference
   summary's `benchmark_implementation` block, fail closed on mismatch,
   both signatures surfaced in the readout (review amendment,
   pre-freeze). The block is labeled `cross_seed_confound: true` —
   different seed AND different budget remain as confounds; a movement
   reading, never a causal isolation.
3. The goal gate recorded per treated arm (strict wins vs base across the
   ten public families; pass = 10).
4. Budget integrity: each arm's `within_budget` flag and wall seconds; if
   any arm exceeds budget, `paired_comparison_valid: false` with the
   reason, scores still recorded.

Preregistered stop outcome (frozen before the event): the pinned gateway
does not emit a soft over-budget flag — an arm that exceeds the tier's
wall budget FAILS with a preserved `budget_gate_failed` failure receipt
and the runner aborts. The line's frozen quick-tier power statements
record that base exceeded budget at tb8192 there, so this is a live risk
at medium. The frozen arm order runs `base` FIRST precisely so a budget
failure spends minimal compute: if any arm fails the gateway's budget
gate, the event closes as `BUDGET_GATE_STOP` — failure receipts and any
completed arms' receipts preserved and reported descriptively, the seed
recorded as spent by the opened ledger record, no retry, no re-run at a
lower budget inside this experiment directory. A successor may re-design
with an intermediate budget under a fresh seed.

Frozen power statement: three SFT pedagogies are dead at the
menders-shaped skill, so a menders flip here relocates the goal venue to
medium@tb8192 and re-prices every successor; a null closes the last cheap
lever and fixes the statechain successor's ceiling at 9/10 honestly. A
goal-gate PASS by any arm would be the program milestone and must survive
the confirmation law before any claim.

## Mandatory checkpoint order

1. Model-free construction + adversarial design review — committed,
   pushed, green.
2. `benchmark` (requires `PASS_BENCHMARK_EVENT` in
   `reports/benchmark_design_review.md`, the design receipt committed at
   HEAD, clean pushed green main). No other stage exists.

## Interpretation limits

One seed, one tier, one budget: readings are a map. The budget contrast
is seed-confounded by construction and is labeled as such; only the
within-event paired comparisons are clean. Benchmark firewall unchanged.
