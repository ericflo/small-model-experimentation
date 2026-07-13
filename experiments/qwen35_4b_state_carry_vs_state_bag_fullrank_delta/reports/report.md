# Report

## Verdict

`PILOT_PROMOTION_BLOCKED` — authoritative post-result disposition. The raw,
immutable analyzer receipt historically emitted `PILOT_STATE_FORMATION_MISS`,
but its classifier did not implement the frozen mutually exclusive taxonomy for
simultaneous failures. Carry failed both non-capacity sign requirements while
also failing state sufficiency, so `capacity_branch_closed=false`. This stops
these checkpoints but does not close the LoRA-rank question.

## Integrity and feasibility

All 11 splits and 27,744 rows match the parent task exactly under both frozen
canonical hashes and direct artifact comparison, with no structural duplicates
or benchmark reads.

On the current 48GB RTX 6000 Ada, G0 discovered the preregistered 62 targets and
892,272,640 FP32 delta parameters. Carry and Bag had identical 892,840,988 total
trainable-parameter/value receipts; every delta tensor received nonzero
gradients in both arms; and Adam allocated 124 finite, shape-matched FP32 moment
tensors. Exact base/K=1 and Carry/Bag parity remained `0.0` before and after the
real optimizer step. K=12 was finite with 682 delta calls per arm. Peak
allocation was 24.49 GiB, peak reservation was 24.93 GiB, and reserved headroom
was 22.57 GiB. The 3.571 GB checkpoint round trip restored recurrent logits
with error `0.0`.

G0 emitted `MODEL_SMOKE_PASS`, proving that the full-rank capacity control was
mechanically executable rather than an arithmetic-only proposal.

## Matched pilot

Carry and Bag independently started from identical 892,840,988-parameter/value
receipts and consumed the same ordered rows, G0 receipt, 300 optimizer steps,
2,594,937 prompt tokens, and 145,316,472 decoder-layer-token applications. Both
checkpoint reloads preserved exact K=1 logits. Peak training allocation was
26.93 GiB in both arms. Final pilot-validation accuracy was 0.28125 for Carry
and 0.328125 for Bag; these were diagnostics, not verdict gates.

The complete locked analysis found:

- primary Carry minus Bag `-0.015625`, 95% CI
  `[-0.06640625, 0.0390625]`, with 3/8 positive depths;
- unseen-K Carry gain `-0.0078125`, CI `[-0.0625, 0.046875]`;
- macro task-mean joint node+phase+checksum step accuracy `0.0027686` versus
  the `0.40` state-sufficiency gate; micro accuracy 7/2,176 = `0.0032169`;
  node step accuracy `0.0616713`;
- joint-holdout Carry minus Bag `+0.01953125`, CI
  `[-0.0234375, 0.0625]`;
- donor-follow gain under 128 bidirectional swap directions `-0.0078125`, CI
  `[-0.0390625, 0.015625]`, with donor follow minus recipient preservation
  `-0.078125`;
- Carry answer-interface validity `0.97265625`; the required +0.05 answer gain
  was mathematically reachable from the Bag baseline `0.32421875`.

The check vector was therefore simultaneous:
`joint_state_sufficient=false`, `positive_carry_minus_bag=false`, and
`query_kinds_positive=false` (node difference `0.0`, checksum difference
`-0.03125`). Preregistration assigns a complete pilot with any non-capacity
promotion failure to `PILOT_PROMOTION_BLOCKED`; the state-specific label is
reserved for a state miss without those unrelated failures. The historical
classifier reversed that precedence. See `terminal_science_audit.md`.

The state was unreadable under the registered macro metric, and pilot swaps did
not support donor following. These are useful descriptive negatives, not G3:
seeds 7411–7413, the same-checkpoint edge cut, and G3 were never run. The phrase
"causally inert" is therefore not licensed by this one-seed diagnostic.

## Capacity interpretation and next experiment

The successor proves that the direct full-shape branch fit, optimized, and also
failed under this 300-step seed-7401 recipe. It does not isolate rank. PEFT LoRA
and the 892M direct-delta construction consume different random streams before
the shared state modules and dropout, so the same integer seed does not give a
bit-identical cross-experiment initialization. Full matrices also change Adam
and global-clipping geometry even when the nominal optimizer schedule is held.

A fresh capacity adjudication is mandatory rather than another stage from these
checkpoints. It must use fresh procedural evaluation rows, exact shared loop-
state initialization and controlled CPU/CUDA/dropout RNG receipts across LoRA
and direct-delta arms, an early held-out trained-depth state positive control,
and fixed-final multi-seed training. It must assign representation formation and
downstream answer/mechanism promotion to separate verdict axes. The current
unreadable state does not license an interface successor.
