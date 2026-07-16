# Preregistration: Clean Gym-Mix Dose

Frozen before any model event. A failed gate is a preserved result, never
permission to change this contract inside this experiment directory.

## Frozen identities

- Experiment: `qwen35_4b_clean_gym_mix_dose`.
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent (baseline and adapter base): the zero_root composite (tree
  414f5829…), authenticated against its lineage merge receipt; the
  retired prefix appears nowhere and its absence is enforced fail-closed.
- Arms: `replay_ctl5` (control, first) and `gym_mix` (candidate); fresh
  rank-32/alpha-64 adapters, training seed 79, standard recipe (190
  updates, LR 1e-5, think/close 0.2/0.2, maxlen 4,096).
- Seeds: construction 77,180; namespace 55,160; gate 88,046; retention
  screens 88,048/88,050/88,051 (frozen 88,047/88,049 collided with a
  predecessor's seeds and were substituted by the next-free rule,
  recorded fail-closed in both receipts); sealed aggregate 78,161.

## Frozen treatment corpus

`data/sft_gym_mix.jsonl`, sha256
`6295011622096992e889b58a1a004fee26f4f9787bd952d348c0bf8593564a89`,
160 rows, regenerates byte-identically; three kinds on invented
executable surfaces (owner-directed design: recreate the retired
prefix's gym-era strengths as fresh documented content):

- 60 `u_siren_episode` (surface stillroom): multi-lookup retrieval
  episodes where 45 rows carry an embedded adversarial imperative with a
  format-matched decoy (decoy ≠ truth per-row verified; obeying the
  imperative is parseable-but-wrong) and 15 rows are clean; think targets
  narrate content-versus-instruction discrimination.
- 50 `u_statechain`: fresh instances from the proven 3-for-3 machinery
  (four formalisms, the byte-copied reviewed generator and validator).
- 50 `u_mirage_abstain` (surface counterhouse): small constraint systems
  with exhaustive-enumeration proofs — 25 uniquely-forced (answer = the
  value) and 25 not-forced (13 unsatisfiable + 12 underdetermined;
  answer = the invented abstain token NOWHERE, which the retention
  evaluator counts as an abstention, never a graded answer); the two
  classes are surface-indistinguishable (identical class token sets,
  audited on every split including unequal holdout splits — a review
  amendment; digit-pair audits; independent brute-force re-derivation in
  tests).

Banned-vocabulary audit extended with twenty description nouns
(injection/retrieval/document/directive/abstain/constraint/impossible
etc.); 34 invented tokens × 69 pinned sources → zero hits; zero row
overlap across all predecessor corpora, streams, and gates.

## Frozen exposure, gates, and event

Standard exact zero-delta MILP (per-arm forward 1,359,192 / nonzero
567,805 / mass×5 621,517; 1,280 aligned core rows; zero skips).
Promotion: candidate axis total (40 rows: 14/13/13 per kind) strictly >
parent AND > replay; at least TWO of the three kinds individually strict
over both controls (ties fail a kind); pooled_k3 retention bands
(−15/+9/−9 sums) vs both controls. Non-promotion seals 78,161.
Conditional benchmark: medium, tb 1,024, sealed 78,161, four arms (base,
zero_root parent, replay_ctl5, gym_mix), the six-slot normalized-pin
hardened runner. Pilot gates: candidate aggregate > base AND > replay
AND > parent. Recorded either way: the goal gate, the per-family table,
and the three AXIS READINGS (candidate vs parent on sirens, rites,
mirage — does fresh gym-style content recover the retired prefix's
margins on clean ground?). Menders is closed; the winnable ceiling is
9/10; any 10/10 is a draw and feeds a fresh confirmation.

## Mandatory checkpoint order

Construction+review → train-control (PASS_CONTROL_TRAINING) →
train-candidate (+committed control receipt) → merge-arms
(PASS_CONTROL_MERGE) → local (PASS_LOCAL_EVENT) → conditional benchmark
(PASS_BENCHMARK_EVENT + promotion receipt). Every stage: clean pushed
green main, byte-identical prerequisites at HEAD, fail-closed pins.

## Interpretation limits

One sealed seed bounds the family readings; the conversion law says
substrate matters and this dose TESTS whether designed substrate
substitutes for inherited substrate — either outcome extends the law.
Benchmark firewall unchanged.
