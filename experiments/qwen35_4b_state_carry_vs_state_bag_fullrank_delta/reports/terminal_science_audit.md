# Terminal Science Audit

## Authoritative disposition

`PILOT_PROMOTION_BLOCKED`, with `capacity_branch_closed=false`.

The immutable `analysis/summary.json` historically emitted
`PILOT_STATE_FORMATION_MISS` and remains preserved byte-for-byte under SHA256
`2f3508202b08928aa6cd2867656e82b6f54859c3c1b075fdb373daa4a2cffa83` and receipt
identity `7697cf03066ff00e41ef02bb0bd3a33b24b42e465106c2db7e474d5f860a0dc0`.
This audit does not rewrite that post-result artifact. It corrects the scientific
disposition by applying the verdict taxonomy frozen before the run.

## Classifier error

The preregistration makes the pilot outcomes mutually distinct:

- missing registered cells are `PILOT_INCOMPLETE`;
- any complete failure of a non-capacity promotion requirement, or an
  unreachable answer gate, is `PILOT_PROMOTION_BLOCKED`;
- only a complete, reachable pilot that specifically fails joint-state
  sufficiency is `PILOT_STATE_FORMATION_MISS`; and
- only `PILOT_PROMOTION_READY` advances.

The realized check vector was complete and reachable but had three failures:

- `joint_state_sufficient=false`;
- `positive_carry_minus_bag=false`; and
- `query_kinds_positive=false`.

Carry minus Bag was `-0.015625`, with 95% pilot interval
`[-0.06640625, 0.0390625]`. The node-query difference was exactly `0.0`; the
checksum-query difference was `-0.03125`. The last two are non-capacity
promotion requirements. The analyzer incorrectly let the simultaneous state
failure take precedence and set `capacity_branch_closed=true`. Under the frozen
taxonomy, their presence requires `PILOT_PROMOTION_BLOCKED` and prohibits the
capacity conclusion.

The stop before seeds 7411–7413 remains correct. The error concerns what the stop
means, not whether these checkpoints may advance.

## Verified raw evidence

The mechanics and evidence bundle are otherwise internally sound:

- exact parent-row parity covered 11 splits and 27,744 rows;
- G0 passed 62 direct targets, 892,272,640 FP32 delta parameters, complete Adam
  state, exact K=1 parity, finite K=12, and behavioral checkpoint restoration;
- Carry and Bag each completed 300 steps with identical within-successor
  initialization, ordered rows, 2,594,937 prompt tokens, and 145,316,472
  decoder-layer-token applications;
- each evaluation contains 768 exact paired rows, and Carry contains all 128
  directed swaps over 64 pairs; and
- independent recomputation reproduced every reported point estimate and
  interval.

The analyzer's `0.002768618` state value is a macro average: it first computes
joint-step accuracy within each task and then weights all 256 tasks equally.
The literal micro count is 7 jointly correct states among 2,176 registered
steps, or `0.003216912`. Both are chance-like and far below `0.40`; the wording
must distinguish them, although this distinction does not affect the blocked
disposition.

The pilot swaps reduced donor following by `0.0078125`, with interval
`[-0.0390625, 0.015625]`. They are diagnostic G1 evidence. No same-checkpoint
edge cut, confirmation seeds, or G3 causal-identification bundle was run, so
this experiment cannot claim completed G3 or general causal inertness.

## Why rank remains unresolved

The direct-delta branch removed the mathematical rank-32 constraint and was
mechanically trainable, but the parent and successor were not a one-factor
randomized rank comparison.

First, the PEFT LoRA and direct-delta constructors consume different random
streams before the shared state initializer, step projection, sufficiency
heads, and training dropout. Constructing direct linear modules and then
zeroing them does not rewind those streams. The same integer seed therefore did
not produce a bit-identical cross-experiment shared initialization or dropout
sequence. Carry and Bag were identical within this successor; that does not
establish parent/successor identity.

Second, replacing 16.8M trainable parameters with 892.8M changes Adam and global
gradient-clipping geometry. Holding the nominal learning rate, warmup, and step
count fixed does not isolate representational rank from optimization dynamics.
Late state loss also remained near the uniform-class baseline, and the design
had no held-out trained-depth state-readability positive control that proved the
state supervision path was learnable before extrapolation.

The supported descriptive claim is narrow: this complete, mechanically valid,
300-step seed-7401 direct-full-shape recipe also failed to learn the registered
state. It neither proves nor rules out LoRA rank as the sufficient explanation.

## Mandatory fresh adjudication

Do not repair this result after the fact, continue its checkpoints, or
reinterpret its pilot swaps as G3. A fresh experiment must be preregistered and
must:

1. use fresh procedural evaluation seeds;
2. load a bit-identical shared loop-state initialization into LoRA and direct-
   delta arms and receipt that identity explicitly;
3. isolate and reset CPU, CUDA, and dropout RNG streams after parameterization-
   specific construction;
4. include an early held-out trained-depth joint-state positive control and
   stop as unresolved if that learning path is not demonstrated;
5. use fixed-final, multi-seed training rather than a single 300-step pilot as
   the terminal capacity estimate; and
6. assign state formation and downstream Carry/Bag, query-stratum, edge-cut,
   and swap evidence to separate verdict axes.

If both parameterizations form state, LoRA demonstrably does not prevent state
formation under the calibrated design. If only direct deltas form state, rank
matters. If neither forms, the experiment must report the state/supervision path
as unresolved rather than converting another simultaneous negative into a rank
conclusion.
