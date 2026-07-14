# Preregistration

## Question and hypothesis

Can a representative exact-token 160- or 240-row designed dose install concise local
execution from replay refresh without requiring the retention-damaging 400-row dose?
The preregistered expectation is that at least one mid-density arm passes the local
gate and retains enough replay policy to beat exact-token replay continuation broadly.

## Frozen data and arms

- Only model: `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Designed source: 800 rows, SHA-256 `4a083375...27c4`.
- Replay source: 2,240 rows, SHA-256 `25a9595f...f0c2`.
- Constructor seed: 77,109.
- Common core: 1,280 replay rows.
- Blocks: three 80-row designed/replay pairs with exact forward-token sums of
  33,613, 34,091, and 33,015.
- Arms: replay-only, designed A+B (160 rows), designed A+B+C (240 rows).
- Per arm: 1,520 rows, exactly 1,405,510 forward tokens, and zero skips.

The representative 320-row arm is explicitly out of scope because exact row-matched
replay parity was infeasible without length-biased designed selection.

## Frozen training

All arms start from the authenticated replay-refresh adapter and use one epoch,
learning rate `1e-5`, rank 32, alpha 64, batch size 1, gradient accumulation 8,
max length 4,096, thought loss weight 0.2, and seed 43. This yields 190 optimizer
steps per arm. Each stage refuses existing outputs and writes a checksum receipt.

## Local promotion gate

Evaluate inherited replay refresh plus all three arms on the same 26 procedural cases
at seed 88,005, greedy decoding, and 1,024 generated tokens. An arm is eligible only
if all conditions hold:

- accuracy ≥0.65;
- parse rate ≥0.90;
- cap contacts ≤2;
- no repeated feasible-route abstention.

Every arm is gated independently. If none passes, write the negative receipts, stop
nonzero, and leave merge and benchmark stages sealed.

## Conditional paired pilot

If at least one arm passes, explicitly merge the replay control and each eligible
designed arm. Consume the single aggregate-only quick@1,024 seed 78,135 event through
the trusted gateway with base, `blend`, inherited replay refresh, replay repeat, and
all locally eligible candidates on the same `qwen_vllm` backend.

A candidate passes the pilot only if it has positive aggregate delta versus base,
strictly positive deltas on all ten reported families versus base, aggregate at least
`blend`, aggregate strictly above inherited replay refresh, and—for a designed
arm—aggregate strictly above replay repeat. Report every registered arm separately.

## Claim boundary

A local pass is installation evidence only. A paired pilot pass is not a universal
claim. Confirmation requires a new experiment with independent quick seeds,
medium@2,048, paired uncertainty, and matched-compute sample-more. No benchmark item,
source, transcript, or private result detail may be read.
