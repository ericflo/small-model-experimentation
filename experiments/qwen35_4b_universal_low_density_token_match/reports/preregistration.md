# Preregistration

Frozen on 2026-07-13 before any adapter in this experiment was trained.

## Arms

- `replay_repeat`: 1,440 shared replay + replay blocks A and B.
- `designed40`: 1,440 shared replay + designed block A + replay block B.
- `designed80`: 1,440 shared replay + designed blocks A and B.

Every block has 40 rows. Replay A equals designed A at 16,732 forward tokens; replay B
equals designed B at 16,543. Every arm has 1,520 rows and 1,429,053 forward tokens.
Designed A and the combined A+B dose each cover all 13 registered skills.

## Training

All arms warm-start the authenticated parent `replay_refresh` adapter. Freeze one
epoch, learning rate `1e-5`, rank 32 / alpha 64, batch 1 x accumulation 8, max length
4,096, `w_think=0.2`, seed 43, and 190 optimizer steps. Any skip, nonfinite loss,
incomplete adapter, hash mismatch, or configuration drift invalidates that arm.

## Local eligibility

Evaluate parent anchor plus all three trained arms greedily on the fresh 26-task,
13-skill synthetic screen at seed 88,004 and max-new-tokens 1,024. Each new arm is
independently eligible only if:

- parse rate is at least 0.90;
- exact accuracy is at least 0.65;
- cap contacts are at most two; and
- it does not abstain on both feasible route tasks.

Local scores do not rank or select among eligible arms. If none passes, benchmark seed
78,134 remains sealed. `replay_repeat` is eligible as a curriculum in its own right;
it remains the mechanism control whenever a designed arm is eligible.

## Aggregate pilot

Run one aggregate-only Menagerie quick@1,024 event at seed 78,134 through the trusted
gateway. Include base, C53 `blend`, inherited replay refresh, replay repeat, and every
locally eligible designed dose, all as explicitly merged checkpoints on `qwen_vllm`.

An arm passes only if it:

1. has positive aggregate delta versus base;
2. is strictly positive versus base on all ten public families;
3. meets or exceeds `blend` aggregate;
4. strictly beats inherited replay-refresh aggregate; and
5. for a designed arm, strictly beats exact-token `replay_repeat` aggregate.

Ties fail the all-family and strict-anchor requirements. No post-event threshold,
family weighting, dose interpolation, or extra arm may be added here.

## Confirmation and claim boundary

A pilot pass is not a universal-feature claim. Move the frozen winner to a new
experiment for independent quick seeds, medium@2,048, paired uncertainty, and a
matched-compute sampling baseline. Require strict family gains to replicate. Preserve
all failures and controls before designing another dose.
