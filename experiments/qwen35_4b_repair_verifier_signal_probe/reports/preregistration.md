# Preregistration: Repair-Verifier Signal Probe

Frozen before any model event. Eval-only feasibility gate for a possible
on-policy episode charter: no training, no promotion, no benchmark seed.

## Frozen identities

- Experiment: `qwen35_4b_repair_verifier_signal_probe`.
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Evaluated composite: `hygiene_explore` (tree 9eb653d7…, weights
  e2112344…, committed merge receipt 22a22a68…), tree-recomputed at every
  boundary; the cell carries the full standalone lineage package
  (six-stage datasets, fixed-seed manifest, vendored trainers/merger and
  root adapter ad2ef4fa…/cd764ae8…, rebuild_lineage verify-inputs wired
  into smoke).
- Probe set: 200 two-alternative items at construction seed 77,160
  (probe source 08856415…, oracle-free runner input 169716c4…, design
  receipt 6fc897f2…), 25 per formalism across the eight reviewed
  legality-bounded machines; overlap receipts zero against the dose-scale
  corpus/holdout, 27 predecessor corpora/streams, and 25 predecessor
  gates.

## The measured signal (post-redesign, frozen)

Each item renders pure failure evidence — the machine spec with legality
clauses, the original written sequence, both trials' setups, WANTED
outcomes, and the OBSERVED outcomes of the broken sequence on both
trials — then two candidate one-step changes in identical grammatical
form with no provenance markers (the unique legal both-trials fix versus
a legal trial-one-consistent distractor), then the letter ask. Solving
requires simulating each candidate against both trials: execution-based
fix verification, exactly the self-check reward signal an on-policy loop
would climb. A 33-token provenance-marker audit shows zero hits across
all 200 prompts; the design receipt records a listing-collision artifact
ceiling of 0.5325 — the best any collision-keyed guessing heuristic can
reach, test-pinned below the signal bar. Construction honesty: instances
whose broken sequence coincidentally succeeded on trial two were
excluded by a deterministic oversample-and-filter (pool 320, exclusions
counted per formalism); position balance is exactly 100 A / 100 B.

## Frozen event and readings

Two sequential authenticated engine runs of the same composite — `think`
(natural thinking, 1,024-token cap) and `nothink` (runner-native
suppression, per-row channel contract) — 400 judgments total, greedy,
runner seed 77,160. Readings (no promotion; exit 0 on any complete
event): per-arm 2AFC accuracy with exact Clopper–Pearson 95% CI;
per-formalism accuracy; position-bias check; per-arm cap-contact counts.

## Frozen consequence partition (ordered, total, two states)

- SIGNAL_PRESENT iff think-arm accuracy ≥ 0.65 (130/200) AND the 95% CI
  excludes 0.5: execution-based fix verification exists for the skill
  generation cannot produce (the C29-class dissociation) — the on-policy
  episode charter is fundable and gets its own intake.
- SIGNAL_ABSENT otherwise: the skill lacks even read-only verification
  signal at this instrument; the on-policy class closes for menders and
  the program map completes at demonstrated-not-confirmed. Scope
  annotation (frozen): if think-arm cap contacts exceed 20% of items
  (strictly greater than 40/200), SIGNAL_ABSENT carries the annotation
  "possibly budget-limited at the 1,024-token cap"; the annotation never
  applies to SIGNAL_PRESENT and never creates a third state.

The nothink arm is descriptive (the C47 substrate-scoping check) and
feeds no consequence.

## Mandatory checkpoint order

1. Model-free construction + review — committed, pushed, green.
2. `local` (requires `PASS_LOCAL_EVENT` in
   reports/local_design_review.md + committed design receipt + clean
   pushed green main). No other stage exists.

## Interpretation limits

One composite, one instrument, 200 items: the probe prices the on-policy
charter's gate, not the charter itself. The distractor class (trial-one-
consistent legal fixes) is the on-policy loop's hardest confusion but not
its only one. Benchmark firewall unchanged.
