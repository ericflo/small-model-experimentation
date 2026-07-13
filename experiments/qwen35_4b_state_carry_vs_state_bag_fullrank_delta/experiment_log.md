# Experiment Log

## 2026-07-13 — successor scaffold and adversarial review

- Created as the parent LoRA pilot's preregistered full-rank capacity successor.
- Copied the self-contained task, recurrent mechanics, evaluation, and analysis
  harness; no parent source is imported at runtime.
- Replaced PEFT with 62 zero-initialized FP32 full-shape deltas on layers 12–19,
  active only for extra R calls.
- Added a strict parent-trigger reader, canonical parent-row parity contract,
  real Adam-state/memory G0, independent Carry/Bag K=1 call checks, and an
  observable delta-plus-loop checkpoint/logit round trip.
- Removed G4 from the CLI and verdict; it remains explicitly deferred.
- Ran CPU unit/static tests only. No data-preparation or model-bearing stage was
  run; there are no scientific results.

## 2026-07-13 — canonical data and live G0

- Canonical CPU smoke passed under config digest
  `bb0abb85766c0e5eb848492a503b1db0e5c005b5d6521e554a3c30d25d514ccd` and
  source contract `c18c44fe8ed6c65fe18be6592ded644a788954a5002256a1dd1730c1fdc8bcba`.
- Regenerated all 11 parent-matched splits (27,744 rows). Frozen canonical-row
  hashes and direct comparison with the available parent artifacts both passed;
  structural duplicates and benchmark reads were zero. Manifest SHA256:
  `1ad19fd3e74e43c52d7e9dc1fbdfc3d9ea0ac4f2b697f6e7e4f7454a40281da5`.
- Live G0 loaded only the pinned `Qwen/Qwen3.5-4B` revision on the RTX 6000 Ada
  and emitted `MODEL_SMOKE_PASS` (receipt identity
  `0832423e632a5c056e701eacb5b7e70387595956cccbadbe9453cb583c8346fc`).
- Exact receipts: 62 targets, 892,272,640 delta parameters, zero initial delta,
  both-arm nonzero gradients, 124 complete FP32 Adam moment tensors, K=1 base
  and Carry/Bag error `0.0` before and after AdamW, and finite K=12 logits with
  682 active delta calls per arm.
- Peak allocation/reservation was 24.49/24.93 GiB with 22.57 GiB reserved
  headroom. The 3,571,392,174-byte checkpoint round trip restored recurrent
  logits exactly and removed its temporary payload.
- These are setup and feasibility receipts only. The seed-7401 Carry/Bag pilot
  is the next authorized scientific stage.
