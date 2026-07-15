# Preregistration: Interleaved-Replay Dose with Medium Pilot

Frozen before any model event. A failed gate is a preserved result; predecessor
failures and sealed seeds untouched.

## Frozen identities

- Experiment: `qwen35_4b_interleaved_replay_dose_medium`.
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent (baseline + warm start): the de-stack experiment's `replay_clean` arm —
  merged tree `19759e12b1301a15a2f9b2db311ffff7c08e3d8b0d3237c9e7cd718fa8dc7f67`,
  weights `2cef3e5e7ddfbee5d2d2c3a878d64f360dc6994764b9856be7e319ce5187d0b4`,
  warm-start adapter weights `f6f910ed1c1dcc843f43e09a562556b8e76ee40096aa7123fd70800d94fc6bb8`,
  config `015bb13568c411c94d24460a9007e1f0d8fe3eb6c9749ad938958490de84d961`.
- Arms: control `replay_interleaved2`; candidate `dose_after_replay`; parent
  label `interleaved_parent`.
- Seeds: inherited corpus construction 77,119 (byte-identical); fresh
  slot/training/gate/aggregate = `55122 / 56 / 88019 / 78149`. 78,149 sealed
  until promotion.

## Frozen treatment corpus (inherited)

Byte-identical inheritance of the independently verified 80-row corpus
(hygiene 40 co-location-hardened / explore 40; 100/100 rows re-derived with
live negative controls in the predecessor's review).

## Frozen exposure match and training

Identical geometry to the predecessor (candidate block = 80 treatment + 160
fillers; control = 240 replay; EXACT three axes; zero skips; encoder bound).
Control first: 1,520 rows, 190 updates, LR 1e-5, rank 32 alpha 64, think/close
0.2/0.2, seed 56, warm start continued in place.

## Frozen gate (seed 88,019)

20-row two-kind axis holdout + 104-row retention screen; overlap receipts
against all six predecessor gates (88,013–88,018) and every corpus/stream;
oracle-free input; documented answer normalization. Promotion identical to the
predecessor: both detectable kinds must strictly win (ties fail; ceiling
exclusion; GATE_UNDETECTABLE fails closed); axis total strictly above both
controls; retention bands correct −5 / caps +3 / parsed −3 versus both; route
abstentions ≤ 4; unconditional recovery flags. No promotion permanently seals
78,149.

Escalation rule (frozen): if the retention bands break DESPITE the interleaved
parent, the dose-vehicle question escalates to a mechanism study; no further
recipe permutations.

## Frozen conditional pilot — MEDIUM tier

One gateway event: tier medium, think budget 1,024, sealed seed 78,149, four
weight-recomputed composites (base `b654e033...`, parent, control, candidate),
clean pushed main with the committed promotion receipt, one-seed ledger,
identical inventory. Gates: candidate aggregate strictly above base, above
`replay_interleaved2`, and above `interleaved_parent`; the
every-family-versus-base record is the goal gate (8-of-92 historical medium
passes; a FAIL is the majority outcome under the hypothesis and is recorded as
"not confirmed at this event").

## Mandatory checkpoint order

1. Model-free construction + design review — committed, pushed, green.
2. train-control; 3. train-candidate (PASS_CONTROL_TRAINING);
4. merge-arms (PASS_CONTROL_MERGE); 5. local; 6. conditional benchmark.

## Interpretation limits

Single-seed events; no claims minted; package-level causal unit; benchmark
firewall unchanged.
