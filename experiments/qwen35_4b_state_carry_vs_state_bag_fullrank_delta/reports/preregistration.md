# Preregistration

## Frozen question

Does replacing the parent's rank-32 extra-R LoRA with direct full-shape deltas
allow a jointly sufficient recurrent state to form, and—if it does—does serial
Carry beat matched-compute reset Bag at unseen semantic depths?

## Frozen design

- Model: pinned `Qwen/Qwen3.5-4B`, Transformers 5.13.0, bf16 base, SDPA.
- Loop: layers 12–19; prelude 0–11 and coda 20–31.
- Added capacity: 62 zero-initialized FP32 direct deltas, 892,272,640 parameters,
  dropout 0.05, scale 2.0, active only on R calls 2..K.
- Same parent substrate and canonical rows; same state modules, losses, K=4,
  training order, pilot seed 7401, full seeds 7411–7413, and evaluation cells.
- Primary comparison: independently trained Carry versus Bag twins, crossed
  task×training-seed bootstrap at K=semantic depth, depths 5–12.
- Mechanism checks: joint trajectory sufficiency, unseen-K gain, joint holdout,
  same-checkpoint Carry edge cut, and bidirectional donor-state swaps.

## Stage gates

- G-trigger: validate the complete parent LoRA deep-state miss.
- G-data: `PARENT_DATA_PARITY_PASS` over every canonical decompressed row.
- G0: live target, exact-path, gradient, real Adam, K=12, memory, and behavioral
  checkpoint-roundtrip pass. Setup-only; no scientific evidence.
- G1: paired seed-7401 pilot. Promote only if every diagnostic cell is complete,
  the answer gate remains reachable, Carry is positive, both query kinds and
  interface checks pass, and joint-state accuracy is at least 0.40.
- G2: fixed-final seeds 7411–7413 with no checkpoint selection.
- G3: edge-cut and counterfactual-swap causal identification.
- G4: deferred to a distinct experiment; unavailable here by construction.

## Fail-closed verdict ladder

Pilot outcomes are mutually distinct: `PILOT_INCOMPLETE` means the registered
diagnostic bundle is missing cells; `PILOT_PROMOTION_BLOCKED` means a complete
pilot failed a non-capacity promotion requirement or the answer gate was not
reachable; `PILOT_STATE_FORMATION_MISS` means a complete, reachable pilot
specifically failed joint-state sufficiency; and `PILOT_PROMOTION_READY` alone
licenses G2. Only `PILOT_STATE_FORMATION_MISS` closes the held-fixed LoRA-rank
capacity branch.

`NONCONFIRMATORY_SMOKE_ONLY` → `SETUP_ONLY` → `UNDER_REPLICATED` →
`NO_SERIAL_STATE_ADVANTAGE` → `TRAINED_UNROLLING_ONLY` →
`SERIAL_BUT_STATE_NOT_SUFFICIENT` → `DEPTH_NOT_ROBUST` →
`DEEP_BUT_NOT_CAUSALLY_IDENTIFIED` → `FULLRANK_CAUSAL_DEPTH_POSITIVE`.

A mechanically valid, complete, reachable G1 joint-state miss is the terminal
answer to the LoRA capacity concern for this held-fixed design. A mechanics,
parity, receipt, resource, incomplete-pilot, or unrelated-promotion failure is
not that scientific negative.
