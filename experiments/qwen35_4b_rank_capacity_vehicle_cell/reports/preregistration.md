# Preregistration: Rank-Capacity Vehicle Cell

Frozen before any model event. A mechanism cell: no promotion, no benchmark
seed, no claim; the verdict selects the successor.

## Frozen identities

- Experiment: `qwen35_4b_rank_capacity_vehicle_cell`.
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Training base (via the per-experiment trainer's `--model-path`): the
  `designed_fresh` merged composite — tree
  `93433aa2d5f3f0d6d4540126579c09feee1d8502df702c1563bae28eb7f60255`, weights
  `0a3b89cdf57ed8a73590580489d744319c12b44b60991db55b5baba6f7c27979`. The
  tokenizer stays hub-pinned; `encode_row` is byte-unchanged from every
  predecessor, so exposure receipts remain comparable.
- New arm: `axis160_r64` — FRESH rank-64 / alpha-128 adapter, NO warm start.
  Published comparison arms: `axis160_direct` (rank 32, known −9 retention;
  committed merge receipt in the dose-diversity cell) and `clean_parent`.
- Seeds: inherited corpus construction 77,117; fresh slot/training/gate =
  `55124 / 58 / 88021`. No aggregate seed exists.

## Frozen treatment corpus and exposure

Byte-identical inheritance of the twice-verified 160-row v1 axis corpus
(`e7a95d73...`); standard exact-exposure stream geometry at slot seed 55,124
(candidate stream only; notional control recorded); zero skips; encoder bound.
Training: 1,520 rows, 190 updates, LR 1e-5, rank 64 alpha 128 (scale 2.0
preserved), think/close 0.2/0.2, seed 58.

## Frozen gate (seed 88,021) and verdict

Instruments as in every predecessor (40 axis at 10 per v1 kind + 104
retention); three weight-authenticated arms; normalization unchanged; overlap
receipts across all eight predecessor gates and every inherited corpus/stream.

Ordered total verdict partition (retention-correct deltas vs `clean_parent` on
this same fresh screen):

1. `SCREEN_INSTABILITY` iff `axis160_direct` ≥ −5 (its −9 fails to reproduce —
   no vehicle inference from this screen) → successor: retention-screen
   calibration study.
2. else `CAPACITY_SUPPORTED` iff `axis160_r64` ≥ −5 → successor: the rank-64
   vehicle becomes the program's dose vehicle; full proven-install recipe at
   rank 64 with the standard gate and the medium pilot behind it.
3. else `CAPACITY_REFUTED` → successor: loss-weighting/update-count cells or
   the preregistered priced-trade gate path.

Secondary readings: `install_preserved` (r64 axis total ≥ r32's on this
screen), per-kind counts (hygiene as the seven-for-seven probe), caps/parse.

## Mandatory checkpoint order

1. Model-free construction + design review — committed, pushed, green.
2. train-candidate (PASS_CONTROL_TRAINING); 3. merge-candidate; 4. local
   (PASS_LOCAL_EVENT). No benchmark stage exists.

## Interpretation limits

Single-seed mechanism cell; the trainer's `--model-path` and (if used) the
local merger variant are per-experiment copies, sha-pinned in receipts;
benchmark firewall unchanged.
