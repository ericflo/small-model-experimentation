# Preregistration

## Question and hypothesis

Can autonomous-close loss placement cross the designed160 local installation gate?
The preregistered expectation is that `close_xi` improves parseability and cap
behavior over byte-identical `standard_xi`, while both are distinguished from an
exact-token replay continuation.

## Frozen data and arms

- Only model: `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: authenticated published `designed160` adapter from
  `qwen35_4b_universal_mid_density_token_match`.
- Designed source: 800 rows, SHA-256 `4a083375...27c4`.
- Replay source: 2,240 rows, SHA-256 `25a9595f...f0c2`.
- Constructor seed: 77,110.
- Common block: 200 replay rows and 199,360 forward tokens.
- Target block: 40 fresh `u_execute` plus 40 fresh `u_induct` rows, zero overlap
  with the parent's 160 designed rows, and 50,726 forward tokens.
- Target filler: 40 replay rows and 36,728 forward tokens.
- Replay-control block: 120 replay rows and exactly 87,454 forward tokens, equal to
  target plus filler.

The three arms are:

- `replay_repeat`: common replay plus replay-control block, ordinary weights;
- `standard_xi`: common replay plus target and filler, ordinary weights;
- `close_xi`: byte-identical `standard_xi` data, with only the target rows'
  autonomous close assigned weight raised from 0.2 to 1.0.

Each file contains 320 trainable rows and exactly 286,814 forward tokens. All rows
fit max length 4,096. The close span encodes to two tokens, so the treatment raises
160 target close tokens from weight 0.2 to 1.0; the other 480 close tokens remain at
0.2. Standard and treatment use identical bytes and ordering.

## Frozen training

All arms start from the same parent and run one epoch, learning rate `1e-5`, rank 32,
alpha 64, batch size 1, gradient accumulation 8, max length 4,096, thought weight
0.2, ordinary close weight 0.2, and seed 44. This gives exactly 40 optimizer steps.
The wrapper refuses overwrites, authenticates the parent and token receipt, and
records logs, package versions, hashes, loss, wall time, and resulting adapter hashes.

## Local promotion gate

Evaluate the immediate parent and all three continuations on the same 26 fresh
procedural cases at seed 88,006 with greedy decoding and a 1,024-token generation
cap. Preserve every completion and summary. Independently gate each trained arm at:

- accuracy ≥0.65;
- parse rate ≥0.90;
- cap contacts ≤2;
- no repeated feasible-route abstention.

Only `standard_xi` and `close_xi` are promotion candidates. If neither passes, write
the negative promotion receipt, stop nonzero, and leave merge and aggregate seed
78,136 sealed. Replay results remain an active-control outcome even if replay itself
passes.

## Conditional paired pilot

If either candidate passes, explicitly merge the immediate parent, replay control,
and every eligible candidate. Consume one aggregate-only quick@1,024 seed 78,136
event through the trusted gateway. The paired model set is base, `blend`, inherited
replay refresh, immediate designed160 parent, active replay continuation, and every
eligible candidate, all on `qwen_vllm`.

A candidate passes the pilot only if it has positive aggregate delta versus base,
strictly positive deltas on all ten reported families versus base, aggregate at
least `blend`, and aggregate strictly above replay refresh, the immediate parent,
and active replay. Report both registered candidates separately when both qualify.

## Claim boundary

A local pass is mechanism evidence only. A paired pilot pass is exploratory and does
not establish generalized installation. Confirmation requires a new experiment with
independent quick seeds, medium@2,048, paired uncertainty, and matched-compute
sample-more. No benchmark item, source, transcript, or private result detail may be
read.
