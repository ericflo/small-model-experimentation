# Preregistration: Hygiene-Explore De-stacked Dose with Medium Pilot

Frozen before any model event. A failed gate is a preserved result; predecessor
failures and sealed seeds are untouched.

## Frozen identities

- Experiment: `qwen35_4b_hygiene_explore_destack_medium`.
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent (baseline + warm start): the `designed_fresh` arm — merged tree
  `93433aa2d5f3f0d6d4540126579c09feee1d8502df702c1563bae28eb7f60255`, weights
  `0a3b89cdf57ed8a73590580489d744319c12b44b60991db55b5baba6f7c27979`, warm-start
  adapter weights `36f41095c2d628e4706694e7d64d16aba815870a1d3660af0e24b14dc0e6b442`,
  config `5966461bd9dbfe280b226940e79eec49030cc44e4131089147634df549dc4055`.
- Arms: control `replay_clean`; candidate `hygiene_explore`; parent label
  `clean_parent`.
- Seeds: construction/slot/training/gate/aggregate =
  `77119 / 55121 / 55 / 88018 / 78148`. 78,148 sealed until promotion.

## Frozen treatment corpus

80 rows from the byte-identical v2 generator at seed 77,119: `u_hygiene` 40
(the co-location-hardened lesson; injection/answer co-location oversampled) and
`u_explore` 40 (unchanged from four measurement events). Executable truth,
uniqueness audits, and the banned-vocabulary scan as previously adversarially
verified.

## Frozen exposure match and training

1,280 shared position-aligned replay rows + one 240-row variable block per arm
(candidate = 80 treatment + 160 fillers; control = 240 replay), EXACT on the
three axes (MILP; infeasibility stops); zero skips; trainer bytes bound.
Control first: 1,520 rows, 190 updates, LR 1e-5, rank 32 alpha 64, think/close
0.2/0.2, seed 55, warm start continued in place (dose two on this lineage; the
interference law bound is dose three).

## Frozen gate (seed 88,018)

Instrument A: 20 axis-holdout rows (10 per kind, fresh seed). Instrument B: 104
retention rows (8 per original skill). One event, three composites, pinned
geometry, oracle-free input, overlap receipts against every predecessor gate
(88,013–88,017) and corpus. Answer normalization identical to v2 (documented;
pre-normalization grades preserved).

Promotion — `hygiene_explore` promotes iff ALL hold: (1) detectability as
before (either control ≥ 9/10 excludes a kind; zero detectable kinds fails
closed); (2) axis total strictly above BOTH controls; (3) strict kind wins on
⌈2/3 × detectable⌉ — with both kinds detectable, BOTH must win; ties fail;
(4) retention bands: correct ≥ each control − 5, caps ≤ each control + 3,
parsed ≥ each control − 3; (5) route abstentions ≤ 4. The receipt records
unconditional recovery flags (`hygiene_win`, `explore_win`) — the de-stacking
reading, adjudicating interference versus content decay regardless of
promotion. No promotion permanently seals 78,148.

Escalation rule (frozen): if EITHER previously-replicated install fails to
recover on this clean lineage, the dose-vehicle question escalates to a
mechanism study (adapter capacity/optimization); no further dose permutations.

## Frozen conditional pilot — MEDIUM tier

One gateway event: tier medium, think budget 1,024, sealed seed 78,148, four
weight-recomputed composites (base `b654e033...`, parent, control, candidate),
clean pushed main with the promotion receipt committed, one-seed ledger,
identical inventory. Gates: candidate aggregate strictly above base, above
`replay_clean`, and above `clean_parent`. The every-family-versus-base record
is the goal gate (8-of-92 historical medium-tier passes; a FAIL is the majority
outcome under the hypothesis and is recorded as "not confirmed at this event").

## Mandatory checkpoint order

1. Model-free construction + design review — committed, pushed, green.
2. train-control; 3. train-candidate (PASS_CONTROL_TRAINING);
4. merge-arms (PASS_CONTROL_MERGE); 5. local; 6. conditional benchmark.

## Interpretation limits

Single-seed events; no claims minted; package-level causal unit; benchmark
firewall unchanged.
