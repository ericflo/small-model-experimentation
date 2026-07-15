# Preregistration: Dose-Diversity Mechanism Cell

Frozen before any model event. This is a mechanism study: no promotion, no
benchmark seed, no claim. The verdict field selects the line's successor.

## Frozen identities

- Experiment: `qwen35_4b_dose_diversity_mechanism_cell`.
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent (warm start + eval label `clean_parent`): `designed_fresh` — tree
  `93433aa2d5f3f0d6d4540126579c09feee1d8502df702c1563bae28eb7f60255`, weights
  `0a3b89cdf57ed8a73590580489d744319c12b44b60991db55b5baba6f7c27979`, adapter
  weights `36f41095c2d628e4706694e7d64d16aba815870a1d3660af0e24b14dc0e6b442`.
- New arm: `axis160_direct`. Published comparison arms:
  `hygiene_explore_direct` (the known −10 retention dose) and `replay_clean`,
  both from the de-stack experiment with committed merge receipts.
- Seeds: inherited corpus construction 77,117; fresh slot/training/gate =
  `55123 / 57 / 88020`. No aggregate seed exists.

## Frozen treatment corpus (inherited)

Byte-identical inheritance of the 160-row v1 axis corpus
(`e7a95d73c619e7c4f20f18ae98ac193e2f57373bd49dc9aede11fd548831686e`; 40 rows
each of tracefix/explore/hygiene/protocol), independently re-derived 200/200 in
two prior reviews. The closed-axis kinds (tracefix, protocol) appear here as
DOSE COMPOSITION, not install claims — the cell measures retention, and
hygiene (six consecutive kind wins) rides as the install probe.

## Frozen training and exposure

One arm: the standard exact-exposure stream geometry (160 treatment + 80
fillers in the 240-row variable block over the 1,280-row core; the matched
control block is solved and recorded for bookkeeping but no control arm
trains). 1,520 rows, 190 updates, LR 1e-5, rank 32 alpha 64, think/close
0.2/0.2, seed 57, warm start continued in place, zero skips, encoder bound.

## Frozen gate (seed 88,020) and the three-way verdict

Instruments: 40-row v1-kind axis holdout (10 per kind) + 104-row retention
screen; four weight-authenticated arms; normalization unchanged; overlap
receipts across all seven predecessor gates and every inherited corpus/stream.

Preregistered readings (retention correct deltas versus `clean_parent` on this
same fresh screen):

- `diversity_mechanism = SUPPORTED` iff `axis160_direct` ≥ −5 AND
  `hygiene_explore_direct` ≤ −6 (the known cost reproduces while the diverse
  dose stays in band) → successor: a diverse-dose deployable recipe with its
  own intake.
- `= REFUTED_INTRINSIC` iff `axis160_direct` ≤ −6 (the diverse dose breaks
  too) → successor: a dose-vehicle study (rank/loss-weights/row-count) or
  acceptance of the retention trade.
- `= SCREEN_FORTUNE_SUSPECT` iff `hygiene_explore_direct` ≥ −5 (the −10 fails
  to reproduce) → successor: retention-band re-calibration across seeds before
  any further dose inference.
- Any other combination is reported verbatim as `MIXED` with no successor
  auto-selected.

Secondary readings: hygiene per-kind counts across all four arms (the install
probe); axis totals; caps/parse per arm.

## Mandatory checkpoint order

1. Model-free construction + design review — committed, pushed, green.
2. train-candidate (PASS_CONTROL_TRAINING); 3. merge-candidate; 4. local
   (PASS_LOCAL_EVENT). No benchmark stage exists.

## Interpretation limits

Single-seed mechanism cell; the verdict selects a successor, it does not mint
a claim; benchmark firewall unchanged.
