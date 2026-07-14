# Protocol Amendment 002: From-Base Replay-Union Local Gate

Frozen on 2026-07-13 before the `designed_plus_replay_fast_b1` adapter finished
training and before any local seed 88,002 output or benchmark seed 78,132 score was
generated.

## Reason

The preregistration names the from-base designed-plus-replay arm but did not quantify
its local installability gate. The first batch-2 execution failed at its first optimizer
step with a WSL/CUDA residency error. The exact 3,040-row corpus, seed, learning rate,
epoch count, effective batch 8, and all other training parameters are unchanged in the
batch-1 / accumulation-8 recovery. Only peak-memory geometry changed.

## Frozen gate

Evaluate the completed adapter alone on 26 freshly generated tasks at seed 88,002,
greedy decoding, and a 1,024-token cap. It must achieve all of:

- exact accuracy at least 0.65;
- parse rate at least 0.90;
- at most 2/26 cap contacts;
- fewer than two abstention-like answers (`INSUFFICIENT`, `None`, null, unknown, or
  equivalent) on the always-feasible routing lessons.

If it fails, preserve the arm as a local negative and consume no benchmark seed. If it
passes, explicitly merge it and run one same-backend aggregate-only quick@1,024 event
at fresh seed 78,132 against the pinned reserialized base and immutable `blend`. The
original pilot promotion rule remains positive aggregate with no negative public-family
delta versus base. No score-conditioned retry remains in this experiment.
