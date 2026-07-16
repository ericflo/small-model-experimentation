# Preregistration: Clean-Path Statechain Extension

Frozen before any model event. A failed gate is a preserved result, never
permission to change this contract inside this experiment directory.

## Frozen identities

- Experiment: `qwen35_4b_clean_path_statechain_extension`.
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent (baseline and adapter base): the ZERO-ROOT composite from
  lifecycle 22's documented six-stage replay — tree 414f5829…, weights
  6e9aad25…, authenticated against the byte-identical provenance copy of
  its merge receipt (e906caea…). The C53-era blend root appears nowhere
  in this cell and its absence is enforced fail-closed: the clean chain
  IS the design.
- Arms: `replay_ctl4` (control, trained first) and `statechain_clean`
  (candidate). Fresh rank-32/alpha-64 adapters, no warm start, training
  seed 73; identical hyperparameters to the proven statechain cell.
- Treatment: the statechain cell's frozen 160-row corpus BYTE-COPIED
  (ab6c7845… — fresh instances would change the treatment; the byte copy
  is the controlled choice, and the copied generator regenerates it
  byte-identically). Replay pool 25a9595f… byte-identical. Namespace
  55,150; exposure exact zero-delta (per-arm forward 1,411,833 / nonzero
  591,024 / mass×5 644,424; 1,280 aligned core rows; zero skips).
- Local gate: statechain holdout at seed 88,041 (10 per formalism, fresh
  instances) + three retention screens at 88,042/88,044/88,045 (88,043
  is taken and skipped, documented); overlap receipts across both cells'
  corpora/streams and all predecessor gates 88,013–88,040.
- Conditional benchmark: medium, tb 1,024, sealed fresh seed 78,160,
  four arms in frozen order (base 26d8ee48…/b654e033…, the zero-root
  parent, replay_ctl4, statechain_clean), hardened runner with the
  SIX-slot normalized-hash pin (41c22c54…) and receipt-pinned ledger.

## Promotion (local)

`statechain_clean` promotes iff ALL: (1) axis total strictly > parent
AND > replay_ctl4 (ties fail); (2) pooled_k3 retention bands on pooled
sums (−15/+9/−9) vs BOTH controls. No floors. Non-promotion permanently
seals 78,160.

## Frozen benchmark readings

Pilot gates (all required): candidate aggregate strictly > base, >
replay_ctl4, > parent. Recorded either way: the goal gate and full
per-family table; the RITES-CONVERSION reading (candidate rites versus
parent and replay, paired — does the program's proven converter
replicate on clean ground); the fully-documented model's profile. Frozen
power statement: menders is closed by rule, so the winnable ceiling is
9/10; any 10/10 is a menders draw and feeds a fresh confirmation cell
before any claim.

## Mandatory checkpoint order

1. Model-free construction + review — committed, pushed, green.
2. train-control (PASS_CONTROL_TRAINING in reports/compute_review.md);
3. train-candidate (+ committed control receipt); 4. merge-arms
(PASS_CONTROL_MERGE in reports/local_design_review.md); 5. local
(PASS_LOCAL_EVENT); 6. conditional benchmark (PASS_BENCHMARK_EVENT in
reports/benchmark_design_review.md + committed promotion receipt).

## Interpretation limits

One seed at the benchmark bounds the conversion reading; the install has
replicated once before at a thin margin on a different parent. The
standalone package reproduces stages 1–7 from the pinned base alone.
Benchmark firewall unchanged.
