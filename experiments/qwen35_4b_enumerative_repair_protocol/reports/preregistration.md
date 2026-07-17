# Preregistration: Enumerative Repair Protocol

Frozen before any model event. A failed gate is a preserved result, never
permission to change this contract inside this experiment directory.

## Frozen identities

- Experiment: `qwen35_4b_enumerative_repair_protocol` (lifecycle 26).
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent (baseline and adapter base): the zero_root composite (tree
  414f5829…), authenticated against its lineage merge receipt; no blend
  root exists anywhere in this cell (fail-closed).
- Arms: `replay_ctl6` (control, first) and `enum_repair` (candidate); fresh
  rank-32/alpha-64 adapters, training seed 83, standard recipe (190
  updates, LR 1e-5, think/close 0.2/0.2, maxlen 4,096).
- Seeds: construction 77,190; namespace 55,170; gate 88,052; retention
  screens 88,053/88,054/88,055; sealed aggregate 78,162 — all verified
  grep-fresh in seed contexts; no substitution was required (the known
  taken seeds 88,043/88,047/88,049 are avoided by the frozen sequence).

## The mechanism argument (why the repair kill rules do not bind)

Every failed menders dose taught the model to INFER the right fix
(eliminative inference — closed at every dose 80–800; even 2AFC
verification at chance). This dose teaches something never tried:
SYSTEMATIC ENUMERATION — given failure evidence, propose the legal
single-step candidates one per turn in a frozen canonical order, let
trial feedback decide, stop at first success. Grounded in the repo's
laws: C34 (brute-force search dominates the model's reasoning — a
model-level law), and protocols are the installable class
(hygiene/explore/termination/statechain — the line's only reliable
installs). The benchmark family is a bounded multi-turn episode WITH
rerun feedback, so an enumerator converts turn budget into coverage
without needing the walled inference skill.

THE TRANSFER BET IS BUDGET-LIMITED, AND THE AMBIGUITY IS QUANTIFIED UP
FRONT (analytic perfect-enumerator simulation, model-free, frozen in the
local design receipt and the corpus manifest): on the 40-row holdout a
PERFECT canonical enumerator needs MEAN 30.4 turns from scratch (median
18.5, max 122; 87.5% of episodes need MORE than 10 turns); on the
160-row treatment corpus, mean 31.4 (median 23, max 125; 88.1% > 10).
The family's episode budget is publicly known only as "bounded". Stated
plainly: if that budget is materially shorter than these needs, a
perfectly-installed enumerator converts few or no episodes — so a
menders reading of 0 is ambiguous between "did not install" and "installed
but out-budgeted", and the consequence rules below disambiguate it a
priori via the fidelity readout.

## Frozen treatment corpus

`data/sft_enum_repair.jsonl`, sha256
`c9b539bf8ce894e92efaafb79e445e9962b3eefe8c1a79e458f082efd1a6744d`,
160 rows, ONE KIND `u_enum_repair` (one kind per dose at full
concentration — the design rule hardened by the gym-mix cell),
regenerates byte-identically; 20 rows per formalism across the eight
machine formalisms reused from the menders dose-scale cell via a
byte-identical machinery copy. Each row renders a PARTIAL enumeration
episode: machine spec with legality clauses + a numbered action list
documenting the full bounded grammar in its frozen order; the broken
written sequence; BOTH trials' wanted+observed failure evidence; the
frozen canonical-order statement (byte-identical in every row: step
number ascending, then action-list position — exactly the generator's
enumeration order); the first k canonical candidates already tried with
their observed two-trial outcomes (all failures by construction; k
cycles over 0/1/3/6/10 — first-candidate rows through deep-in-the-list
rows); and the ask: name the NEXT untried legal candidate. Think target:
enumerate the legal set, cross off the tried ones, emit the next. Answer:
exact-match `STEP <k>: <corrected step>` (the predecessor cells' format).

Per-row generator verification (exhaustive re-derivation over the full
single-step candidate space): the target IS the canonical-next untried
legal candidate; exactly ONE candidate repairs both trials; every tried
entry is legal, canonically ordered, and genuinely failing (each
re-simulated against both trials). Banned vocabulary: the menders cell's
full inventory (public family names, blocker-family description nouns,
every prior surface pool), scanned case-insensitively. Surfaces are ALL
inherited (no fresh-surface claim); the load-bearing freshness receipt
is the ROW-overlap audit: zero canonical-user-message overlap against 76
pinned predecessor sources including the formalism-sharing menders
corpus, streams, and gates.

## Frozen exposure, gates, and event

Standard exact zero-delta MILP (per-arm forward 1,436,178 / nonzero
572,724 / mass×5 629,552; 1,280 aligned core rows; zero skips;
infeasibility would have been a preregistered STOP).

Promotion (single-kind): candidate axis total (40 rows at seed 88,052; 5
per formalism, one per k-value; same invariants as the treatment)
strictly > parent AND > replay_ctl6 — ties fail; NO per-kind split
exists; per-formalism correctness is reported descriptively, never
gated; pooled_k3 retention bands (−15/+9/−9 sums) vs both controls.
Non-promotion seals 78,162.

Preregistered NON-GATING mechanism readings, recorded either way:

- `episode_success_simulation` (design receipt, analytic, model-free):
  the number of turns a PERFECT canonical enumerator needs per holdout
  episode (canonical index of the unique both-trials fix + 1, plus the
  remaining count after the rendered tried prefix), with distributions.
- `enumeration_fidelity` (eval receipt): for every axis row, three
  booleans about the model's proposal — (a) LEGAL, (b) UNTRIED, (c)
  CANONICAL-NEXT — the mechanism decomposition beyond raw correctness,
  summarized per arm by the gate.

Conditional benchmark: medium, tb 1,024, sealed fresh seed 78,162, four
arms in frozen order (base 26d8ee48…/b654e033…, zero_root_parent
414f5829…, replay_ctl6, enum_repair), the six-slot normalized-pin
hardened runner (six fail-closed trained-arm TODO-PIN slots covered by
check_design's normalized-hash pin). Pilot gate: candidate aggregate
strictly > base AND > replay_ctl6 AND > zero_root_parent. Recorded
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

Ordering note (honest provenance): the negative scoping and the quoted
simulation numbers were added PRE-FREEZE by review amendment, after the
adversarial review flagged that only the positive consequence had been
frozen; no model event had run and no seed had been consumed.
