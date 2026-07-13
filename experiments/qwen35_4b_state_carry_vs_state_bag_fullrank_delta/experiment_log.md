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

## 2026-07-13 — matched full-rank pilot and historical analyzer output

- Trained independent Carry and Bag seed-7401 arms for the fixed 300 steps.
  Both consumed 2,594,937 prompt tokens and 145,316,472 decoder-layer-token
  applications in identical order (`97813bf9a2c7b81cf55db1a405e8e999e7e4bf953b2d50434a007140019b0e4f`),
  under the same source/config/data/G0 lineage and identical initialization
  receipt. Peak allocation was 26.93 GiB in each arm.
- Carry final validation was 0.28125 and Bag was 0.328125. Both fixed-final
  checkpoints passed exact K=1 reload parity and yielded complete 768-row pilot
  evaluations; Carry also yielded all 128 bidirectional swap directions.
- Deterministic analysis was rerun byte-identically (summary SHA256
  `2f3508202b08928aa6cd2867656e82b6f54859c3c1b075fdb373daa4a2cffa83`)
  and historically emitted `PILOT_STATE_FORMATION_MISS` with receipt identity
  `7697cf03066ff00e41ef02bb0bd3a33b24b42e465106c2db7e474d5f860a0dc0`.
- Macro task-mean joint-state step accuracy was 0.0027686 versus the 0.40 gate;
  the micro count was 7/2,176 = 0.0032169. Carry minus Bag was
  -0.015625 (CI -0.06640625 to +0.0390625), unseen-K gain -0.0078125 (CI
  -0.0625 to +0.046875), and swaps reduced donor following by 0.0078125.
- All required cells were complete, the answer gate was reachable, and the
  answer interface was valid. The emitted check vector nevertheless also had
  `positive_carry_minus_bag=false` and `query_kinds_positive=false` in addition
  to `joint_state_sufficient=false`.
- Confirmation seeds, edge cut, G3, and G4 were not run. No interface successor
  is licensed because the state was not readable.

## 2026-07-13 — post-result terminal science audit

- An adversarial read-only audit found that the analyzer's precedence did not
  implement the frozen mutually exclusive pilot taxonomy. The preregistration
  assigns a complete pilot with any non-capacity promotion failure to
  `PILOT_PROMOTION_BLOCKED`; the implementation instead let the simultaneous
  state failure dominate.
- The authoritative disposition is therefore `PILOT_PROMOTION_BLOCKED`, with
  `capacity_branch_closed=false`. The raw summary, receipt identity, and hash
  above are preserved unchanged as historical flawed-classifier output rather
  than rewritten after seeing the result.
- The audit also found that the parent LoRA and successor direct-delta builds did
  not share bit-identical state-module initialization or dropout RNG streams.
  Their parameterizations consume different random streams before training,
  and the 892M-parameter direct optimizer has different global-clipping geometry.
  The same integer seed, learning rate, and schedule do not remove those
  cross-experiment confounds.
- The result remains strong descriptive evidence that this mechanically valid
  direct-full-shape recipe did not learn the registered state. It does not close
  LoRA rank as the cause or non-cause, and it did not run edge cuts or G3.
- A fresh experiment is mandatory: use fresh procedural evaluation rows,
  bit-identical shared loop-state initialization, isolated and reset RNG streams,
  an early trained-depth state-readability positive control, full fixed-final
  multi-seed LoRA/direct-delta arms, and separate representation versus
  answer/mechanism verdict axes.
