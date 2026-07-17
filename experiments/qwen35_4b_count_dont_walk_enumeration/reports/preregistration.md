# Preregistration: Count-Don't-Walk Enumeration

Frozen before any model event. A failed gate is a preserved result, never
permission to change this contract inside this experiment directory.

## Frozen identities

- Experiment: `qwen35_4b_count_dont_walk_enumeration` (lifecycle 27).
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent (baseline and adapter base): the zero_root composite (tree
  414f5829…), authenticated against its lineage merge receipt; no blend
  root exists anywhere in this cell (fail-closed).
- Arms: `replay_ctl7` (control, first) and `count_walk` (candidate); fresh
  rank-32/alpha-64 adapters, training seed 85, standard recipe (190
  updates, LR 1e-5, think/close 0.2/0.2, maxlen 4,096).
- Seeds: construction 77,191; namespace 55,171; gate 88,056; retention
  screens 88,057/88,058/88,059; sealed aggregate 78,163 — all verified
  grep-fresh in seed contexts. Known-taken: 88,043/88,047/88,049 and
  everything <= 88,055 (including the reference cell's 88,052–88,055);
  benchmark seeds spent through 78,162. ONE next-free substitution
  recorded: training seed 84 (next after the reference's 83) is taken as
  a task seed of `qwen35_4b_hypothesize_verify_wall` and appears in
  `qwen35_4b_meta_induction` per-row data fields, so the frozen training
  seed is 85.

## The evidence (why this cell exists)

The enumerative-repair cell (lifecycle 26,
`qwen35_4b_enumerative_repair_protocol`) proved the enumeration
DISCIPLINE installs: 9/40 canonical-next on its axis holdout versus BOTH
controls at exactly 0/40 — the program's starkest mechanism contrast.
Its committed truncation forensics
(`experiments/qwen35_4b_enumerative_repair_protocol/analysis/truncation_forensics.md`)
showed the failure was EXPRESSION COST, not discipline: 20 of the 21
unparseable gate rows ended at the 1,024-token cap mid-CORRECT walk —
the model faithfully executed the frozen canonical walk, whose token
cost grows with the tried-list depth k and the grammar size, and ran out
of budget on deep-k rows. A verbose walker also cannot fit repair turns
inside a bounded episode, independently of the frozen verdict.

## The one designed delta (expression pedagogy only)

This cell clones the reference cell ENTIRELY — machine simulators,
legality bounding, canonical order, K_CYCLE, uniqueness invariants,
single-kind 160-row dose, exact zero-delta MILP, single-kind promotion,
the frozen two-direction menders consequences with the 0.50 fidelity
precondition, the six-slot normalized-pin hardened runner — and changes
ONLY how the answer is expressed:

1. THINK TARGETS teach COUNT-DON'T-WALK: a fixed-shape compact
   computation, identical structure in every row, exactly five short
   lines — (a) count the tried entries → k; (b) the target is change
   number k+1 in the frozen order; (c) locate k+1 by the rendered range
   arithmetic (find the step whose cumulative range contains k+1, with
   the explicit offset subtraction); (d) resolve the offset to the
   action-list slot, skipping the step's written action; then emit
   `STEP <n>: <action>`. Constant token cost in k. A frozen THINK LENGTH
   BUDGET is enforced per row: five-line shape + character/estimate caps
   in the generator, and the REAL tokenizer bound (<= 120 Qwen tokens
   per think span) fail-closed in `measure_source_tokens.py` (measured:
   max 105, mean 95.8 real tokens over the frozen corpus).
2. THE ORDER STATEMENT gains the rendered per-step candidate counts
   ("step 1 offers 17 changes (numbers 1-17); step 2 offers 17 (numbers
   18-34); …"), byte-identical rule text otherwise; per-step counts are
   computed generically (len(action list) − 1 when the written action is
   in the list, else len(action list)) and the generator verifies the
   rendered ranges against its own exhaustive enumeration exactly, range
   by range.
3. NEW GATE READING (recorded, non-gating): `expression_cost` — the
   per-arm think-token-length distribution over the 40 axis rows plus
   the truncation count, summarized by the gate from every row's
   `n_thinking_tokens` and `cap_contact`. This is the expression-cost
   reading the lineage now owes; it never feeds promotion.

THE TRANSFER BET REMAINS BUDGET-LIMITED, AND THE AMBIGUITY IS QUANTIFIED
UP FRONT (analytic perfect-enumerator simulation, model-free, frozen in
the local design receipt and the corpus manifest): on the 40-row holdout
a PERFECT canonical enumerator needs MEAN 27.1 turns from scratch
(median 20.5, max 78; 80.0% of episodes need MORE than 10 turns); on the
160-row treatment corpus, mean 32.6 (median 22, max 125; 86.9% > 10).
The family's episode budget is publicly known only as "bounded"; the
consequence rules below disambiguate the zero draw a priori via the
fidelity readout.

## Frozen treatment corpus

`data/sft_count_walk.jsonl`, sha256
`21e6f5cb705f447f7a4dfc9bff24673f798f48df312b99a6cf686505855ee096`,
160 rows, ONE KIND `u_count_walk` (one kind per dose at full
concentration — the design rule hardened by the gym-mix cell),
regenerates byte-identically; 20 rows per formalism across the eight
machine formalisms reused from the menders dose-scale cell via a
byte-identical machinery copy. Each row renders a PARTIAL enumeration
episode exactly as the reference cell's (spec + numbered action list +
broken sequence + both trials' failure evidence + the frozen
canonical-order statement + the rendered per-step ranges + the first k
canonical candidates already tried + the ask). Answer: exact-match
`STEP <k>: <corrected step>` (unchanged format).

Per-row generator verification (exhaustive re-derivation over the full
single-step candidate space): the target IS the canonical-next untried
legal candidate; exactly ONE candidate repairs both trials; every tried
entry is legal, canonically ordered, and genuinely failing (each
re-simulated against both trials); the rendered ranges equal the
enumeration exactly; the think target IS the frozen five-line compact
computation (byte-compared against a pure re-derivation) under the
frozen budget. Banned vocabulary: the menders cell's full inventory,
scanned case-insensitively. Surfaces are ALL inherited (no fresh-surface
claim); the load-bearing freshness receipt is the ROW-overlap audit:
zero canonical-user-message overlap against 83 pinned predecessor
sources — including the reference cell's corpus, streams, and its four
gate files (seeds 88,052–88,055), the formalism-sharing menders corpus,
streams, and gates, and the six zero-root lineage datasets.

## Frozen exposure, gates, and event

Standard exact zero-delta MILP (per-arm forward 1,438,010 / nonzero
564,379 / mass×5 621,239; 1,280 aligned core rows; zero skips;
infeasibility would have been a preregistered STOP).

Promotion (single-kind): candidate axis total (40 rows at seed 88,056; 5
per formalism, one per k-value; same invariants as the treatment)
strictly > parent AND > replay_ctl7 — ties fail; NO per-kind split
exists; per-formalism correctness is reported descriptively, never
gated; pooled_k3 retention bands (−15/+9/−9 sums) vs both controls.
Non-promotion seals 78,163.

Preregistered NON-GATING mechanism readings, recorded either way:

- `episode_success_simulation` (design receipt, analytic, model-free):
  the number of turns a PERFECT canonical enumerator needs per holdout
  episode, with distributions.
- `enumeration_fidelity` (eval receipt): for every axis row, three
  booleans about the model's proposal — (a) LEGAL, (b) UNTRIED, (c)
  CANONICAL-NEXT — summarized per arm by the gate.
- `expression_cost` (eval receipt, NEW): per-arm think-token-length
  distribution on the axis rows + truncation count — the reading this
  lineage owes after the reference cell's truncation forensics.

Conditional benchmark: medium, tb 1,024, sealed fresh seed 78,163, four
arms in frozen order (base 26d8ee48…/b654e033…, zero_root_parent
414f5829…, replay_ctl7, count_walk), the six-slot normalized-pin
hardened runner (six fail-closed trained-arm TODO-PIN slots covered by
check_design's normalized-hash pin). Pilot gate: candidate aggregate
strictly > base AND > replay_ctl7 AND > zero_root_parent. Recorded
either way: the goal gate, the per-family table, and THE MENDERS READING
— candidate vs base and vs parent on menders specifically; the frozen
question: does taught enumeration convert to the family with live rerun
feedback?

## Frozen ordered consequences for the menders reading

Menders has been 0-margin on most draws across the program's sealed
events, and the simulation above makes the zero draw ambiguous, so BOTH
directions are frozen a priori, in this precedence order (no third state
exists for the zero draw):

1. POSITIVE (takes precedence): ANY candidate menders > 0 where the
   controls sit at 0 is the mechanism answer, even without promotion of
   the reading itself.
2. TURN_BUDGET_SCOPED: a benchmark menders reading of 0 WITH the
   fidelity precondition met is frozen as TURN_BUDGET_SCOPED —
   "enumeration installed with high fidelity but did not convert within
   the family's episode budget; the protocol-install mechanism is NOT
   refuted; what closes is the pure-enumeration route at the family's
   actual budget."
3. OTHERWISE a menders 0 reads as the install/conversion failing on its
   own terms.

Fidelity precondition (defined numerically a priori, keyed to the
preregistered enumeration-fidelity readout; integer-exact comparisons):
let F = the candidate's CANONICAL-NEXT rate on the 40-row axis holdout,
read from the local promotion receipt's mechanism_reading. The
precondition HOLDS iff the candidate PROMOTED locally AND F >= 0.50 AND
F strictly exceeds BOTH controls' canonical-next rates. Implemented in
`run_benchmark.fidelity_precondition` / `menders_reading`
(frozen_interpretation ∈ {MECHANISM_ANSWER, TURN_BUDGET_SCOPED,
FAILED_ON_ITS_OWN_TERMS}), covered by the normalized-hash code pin.

A 10/10 goal gate is a menders draw and feeds a fresh confirmation cell
before any claim.

Provenance note: the two-direction consequences, the 0.50 fidelity
precondition, and the quoted simulation numbers are INHERITED VERBATIM
in structure from the reference cell's reviewed, frozen contract (where
they were added pre-freeze by review amendment); the numbers quoted here
are this cell's own, computed model-free from its frozen corpus and
holdout before any model event.
