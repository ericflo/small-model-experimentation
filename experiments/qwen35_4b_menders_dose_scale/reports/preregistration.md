# Preregistration: Menders Dose Scale (lifecycle 20)

Frozen before any GPU stage. The design receipt (`data/design_receipt.json`,
`check_design.py --check`) and the local design receipt
(`data/local_design_receipt.json`, `gen_local_gate.py --check`) pin every
constant below; the seed-consuming benchmark runner re-runs both checks at
the boundary.

## Arms

- Parent and adapter base: `hygiene_explore` composite (tree 9eb653d7…),
  fresh rank-32/alpha-64 adapters, NO warm start, training seed 71.
- `replay_ctl3` — control, trains FIRST (1,000 replay slots in its
  variable block; arm-level multiplicity <= 2, repeats solver-minimized).
- `feedloop_scale` — candidate (800 u_feedloop treatment rows + 200 replay
  fillers; duplicate-free).
- Shared: 1,280 position-aligned replay core rows; 2,280 rows/arm; exact
  zero-delta on forward / nonzero-target / absolute-loss-mass-x5; 285
  optimizer updates at accumulation 8; LR 1e-5 cosine, warmup 0.03,
  batch 1, think/close weights 0.2/0.2, answer 1.0, max length 4096.

## Treatment corpus

`data/sft_feedloop_scale.jsonl` — 800 rows (sha 080c3603…), construction
seed 77,150, u_feedloop episodes ONLY, 100 per formalism across eight
legality-bounded formalisms (troughline, trinketcord, crankwheel,
sigilslate reused as fresh instances; barrowyoke, balesled, millround,
skeinreel new). Reviewed invariants: >=2 legal fix candidates after
round-1 evidence with the wrong attempt among them; exactly 1 after
rounds 1+2; extended-grammar exclusion audit with a per-formalism probe
scope recorded row-by-row (numeric parameters to 12; item parameters —
knots, etches, lashes/shoves — over the full module pools; for the two
named-container machines, troughline and barrowyoke, the container
dimension of every op is ADDITIONALLY probed over the full pool via a
tolerant probe apply in which phantom containers start empty; sigilslate
slot indices are the slate's four physical slots, structural rather than
a documented-pool parameter, and are not probed past 4; every extra
survivor excluded by the rendered legality clause alone); think targets
quantify over legal steps; repairs easy by design. Banned-vocabulary
scan extended with the statechain cells' surface pools; fresh-surface +
row-overlap audits vs 36 pinned sources.

## Local gate (seed 88037 + screens 88038/88039/88040)

`feedloop_scale` promotes iff ALL of:

1. axis total (40 fresh u_feedloop rows, 5 per formalism) STRICTLY >
   parent AND > replay_ctl3 — single-kind dose, no per-kind split,
   per-formalism reported never gated;
2. pooled_k3 retention bands on pooled sums vs BOTH controls:
   correct >= reference - 15, cap contacts <= reference + 9,
   parsed >= reference - 9 (i.e. +-5/3/3 on means; exact integer
   arithmetic on sums).

No absolute per-kind floors. No passing candidate keeps aggregate seed
78,158 sealed.

## Preregistered NON-GATING dose-response reading

Recorded either way from the same axis event; never feeds the verdict.
Baseline: the reference cell's candidate (`feedloop_state`) scored
u_feedloop 0/20 on fresh instances at the 80-row dose (promotion receipt
sha d232a1be…). This cell's reading: candidate u_feedloop axis total out
of 40, rendered per formalism. DOSE x DIVERSITY CONFOUND, stated in the
frozen statements: formalism diversity doubled simultaneously with dose
(4 -> 8 formalisms; 20 -> 100 rows per formalism), so a nonzero reading
is NOT a pure dose-response isolate — the 10x dose is the dominant
delta, but diversity moved with it. Frozen consequence statements:

- IF NONZERO: "nonzero fresh-instance u_feedloop transfer at 10x the
  failed dose is evidence that SCALE-PLUS-DIVERSITY reopens the family
  (C43: the 80-row install was data-limited; the 10x dose is the
  dominant delta, but formalism diversity doubled 4->8 with it, so this
  is not a pure dose-response isolate), even without promotion."
- IF ZERO: "a 0 at 10x the failed dose closes the dose-scale mechanism
  class AND the added-diversity variant together for the feedback-loop
  skill on this parent."

## Conditional benchmark (only on promotion)

Medium tier, think budget 1024, ONE sealed fresh seed 78,158, four
composites in frozen order: base (weights b654e033…, tree 26d8ee48…),
hygiene_explore_parent, replay_ctl3, feedloop_scale. Receipt-pinned
closed ledger: the write-ahead opened record spends the seed; the closed
record sha-pins the summary AND all four gateway receipts; a closed
record refuses forever; unopened events demand a clean slate; a crashed
summary must regenerate byte-identically before closing.

- PILOT GATE: candidate aggregate strictly > base AND > replay_ctl3 AND
  > hygiene_explore_parent.
- GOAL GATE: all ten public families strictly > base — recorded either
  way, never part of the pilot pass.
- FROZEN POWER STATEMENT: menders alone gates the all-families goal
  (nine families hold vs base on every sealed seed; the ties are
  0-margin); three small-dose pedagogies failed at it and dose scale is
  the one permitted mechanism class, so menders > 0 for the candidate on
  this seed is the reading of consequence; any 10/10 feeds a fresh
  confirmation cell (independent seeds + matched compute) before any
  claim.

## Stops

- MILP infeasibility at the frozen geometry is a preregistered STOP
  (`stream_manifest.json` outcome `STOP_EXPOSURE_MATCH_INFEASIBLE`).
  The shipped manifest records `PASS_EXPOSURE_MATCH` under the documented
  pool-bind formulation (control block over the full pool, arm-level
  multiplicity <= 2, repeats solver-minimized to 575 at gap 0).
  RESIDUAL BIAS DIRECTION of the control-arm repetition, stated
  explicitly: repetition plausibly DEFLATES the replay control slightly
  (diminishing returns on the twice-seen rows versus fresh replay rows),
  which makes candidate-vs-replay comparisons marginally EASIER for the
  candidate; the PARENT-anchored bars (axis strictly > parent; retention
  bands vs parent) bind independently and are unaffected, so no
  false-promotion pathway opens that the parent bars do not still guard;
  the retention band vs replay is conservative in the direction that
  costs the candidate nothing.
- Any tokenizer skip in either stream aborts training.
- Every stage requires its committed review verdict at a clean pushed
  main HEAD (PASS_CONTROL_TRAINING / PASS_CONTROL_MERGE /
  PASS_LOCAL_EVENT / PASS_BENCHMARK_EVENT).

## Seeds

77,150 (construction), 55,140 (stream namespace), 71 (training), 88,037
(axis gate), 88,038-88,040 (retention screens), 78,158 (sealed
aggregate) — all verified grep-fresh in seed contexts at design time; no
substitution required (the seed-71 audit record lives in the design
receipt).
