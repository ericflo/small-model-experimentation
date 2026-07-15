# Preregistration: Axis Corpus V2 with Staged Repair

Frozen before any model event. A failed gate is a preserved result. Predecessor
failures and sealed seeds are untouched.

## Frozen identities

- Experiment: `qwen35_4b_axis_corpus_v2_staged_repair`.
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent (baseline + warm start): the `axis_on_replay` arm of the stack trial —
  merged tree `77e4858fe6ddade7a8446a0c561c3c18d07c338d4dea2f0b8193693fcca264ea`,
  weights `7ebcad397c820196fb2271fe4c608a62a578465152b48e3fcee2c8d3b46fd0e4`,
  warm-start adapter weights `87cdebde17d6151f440dd5c8fe28abc69ff074036c889eb1e4732775a76f3801`,
  config `3d7a87050162955410220c74fe43131f710057bec9bc5052b66f824322e1b766`.
- Arms: control `replay_repeat3`; candidate `axis_v2`; parent label `axis_parent`.
- Seeds: construction/slot/training/gate/aggregate = `77118 / 55120 / 54 / 88017 / 78147`.
  78,147 stays sealed until promotion.

## Frozen treatment corpus

160 rows from `scripts/gen_axis_v2.py` at seed 77,118: `u_bugfind` 30
(localize-only `STEP <k>`; early bugs oversampled — observed 25/30 in the first
program half), `u_bugmend` 25 (corrected instruction, step given), `u_retrace`
25 (`STEP <k>; <final state>`), `u_explore` 40 (unchanged from the
three-times-replicated install), `u_hygiene` 40 (31 injected, 17 with the
injection co-located in the queried record). Every think demonstrates its
bounded search with explicitly rejected candidates (op-TYPE changes included),
the two checkpoint rules from the forensics, and an immediate canonical commit.
Executable truth for every row; repair/localization uniqueness enforced by
exhaustive enumeration over the visible grammar; banned-vocabulary scan
unchanged from v1 (already adversarially verified).

## Frozen exposure match and training

As in the predecessors: 1,280 shared position-aligned replay rows plus one
240-row variable block per arm (candidate = 160 treatment + 80 fillers;
control = 240 replay), EXACT on the three axes (MILP; infeasibility stops);
zero encoder-skipped rows; trainer bytes bound to the receipt. Control first,
then candidate: 1,520 rows, 190 updates, LR 1e-5, rank 32 alpha 64,
think/close 0.2/0.2, seed 54, warm start continued in place.

## Frozen gate (seed 88,017)

Instrument A: 50 axis-holdout rows (10 per v2 kind, fresh seed); the bugfind rows are position-stratified (five first-half, five second-half bug steps) so the kill-rule kind tests localization rather than the training corpus's deliberate early-position prior; the promotion receipt records the two kill-rule win flags unconditionally. Instrument B:
104 retention rows (8 per original skill). One event, three composites, pinned
geometry, oracle-free input, overlap receipts against every predecessor gate
and corpus.

ANSWER NORMALIZATION (prospective; identical for every arm and instrument;
recorded in the receipt): collapse whitespace runs to single spaces, strip,
then delete spaces adjacent to '>' and ';'. Rationale: 21 correct-but-rejected
whitespace rows measured across the three prior events (forensics document).

Corrected promotion — `axis_v2` promotes iff ALL hold: (1) detectability as in
the re-adjudication (neither control ≥ 9/10; excluded kinds reported;
`GATE_UNDETECTABLE` fails closed); (2) axis total strictly above BOTH controls;
(3) strict kind wins on ⌈2/3 × detectable⌉ kinds, ties fail; (4) retention
bands: correct ≥ each control − 5, caps ≤ each control + 3, parsed ≥ each
control − 3; (5) route abstentions ≤ 4. No promotion permanently seals 78,147.

FROZEN KILL RULE for the line: if neither `u_bugfind` nor `u_bugmend` registers
a strict holdout win over both controls, the trace-repair axis closes for this
model at this dose; no v3 of the same mechanism may be opened without a new
mechanism argument.

## Frozen conditional pilot — MEDIUM tier

One gateway event: tier medium, think budget 1,024, sealed seed 78,147, four
weight-recomputed composites (base `b654e033...`, parent, control, candidate),
clean pushed main with the promotion receipt committed, one-seed ledger,
identical inventory. Gates: candidate aggregate strictly above base, above
`replay_repeat3`, and above `axis_parent`. The every-family-versus-base record
is the goal gate (8-of-92 medium-tier historical passes; a FAIL is the
majority outcome under the hypothesis and is recorded as "not confirmed at
this event"). Secondary frozen readings: per-kind install map for the three
redesigned lessons; hygiene co-location effect (failures on co-located vs
separated injections); normalization effect (rows whose grade changed under
normalization, per arm).

## Mandatory checkpoint order

1. Model-free construction + design review — committed, pushed, green.
2. train-control; 3. train-candidate (PASS_CONTROL_TRAINING required);
4. merge-arms (PASS_CONTROL_MERGE); 5. local; 6. conditional benchmark.

## Interpretation limits

Single-seed events; no claims minted; the package-level causal unit is the
whole arm; per-kind attribution uses the preregistered holdout counts.
Benchmark firewall unchanged.
