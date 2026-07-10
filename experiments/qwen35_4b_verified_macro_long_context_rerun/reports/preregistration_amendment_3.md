# Preregistration amendment 3: periodic-loop termination

Historical note: amendment 4 retained this detector but showed that think@16,384 already gives all
productive traces 3,820 tokens of headroom. It supersedes the 32k operating-point choice to recover
vLLM concurrency before any complete scientific smoke artifact existed.

Date: 2026-07-09. Frozen after the complete train-only calibration exhausted the original ladder
and before the heldout interface gate or any fresh scientific prompt.

## Observation

The 32,768-token tier produced a sharply bimodal termination distribution. Fifty-five of 64
samples closed naturally, all by 12,564 thinking tokens, and no answer truncated. The same nine
samples that contacted 16,384 continued through 32,768.

Because the registered ladder ended setup-inconclusive, generated text from those train-only
calibration records was inspected for interface diagnosis. Every cap-contact tail was an exact
token-periodic loop over the final 8,192 tokens, with periods 5–318 tokens and match rate 1.0. The
tails repeatedly rechecked an already resolved macro substitution; they were not continuing to
accumulate new reasoning. Naturally closing samples had no comparable tail. Frozen evidence is in
`analysis/periodic_loop_audit.json`.

No heldout interface output, fresh smoke task, full task, parser result, or correctness label was
used in this diagnosis.

## Registered detector and branch

At think@32,768 only, classify a cap contact as a periodic loop when all conditions hold:

1. at least 8,192 retained thinking token ids are available;
2. some period from 1 through 2,048 tokens makes at least 99% of the final 8,192-token tail equal
   to the same tail shifted by that period; and
3. the total periodic-loop rate is at most 25%.

The detector operates on token ids and termination metadata, not decoded semantics or correctness.
A proven periodic loop may use the existing two-stage runner's forced close and 512-token answer
stage. It is reported separately from natural termination. Any other cap contact remains unresolved
censoring and retains the original below-5% gate. Answer truncation remains censoring under all
circumstances.

For headroom, compute p99 over naturally closed samples after separating proven loops. The
think@32,768 calibration clears: natural p99 is 12,564 (<24,576), periodic loops are 9/64 (14.1%,
below 25%), unresolved contacts are 0, and answer truncation is 0.

Apply the same frozen detector and thresholds to heldout interface, proposal, smoke, and full
outputs. Every comparison arm still receives think@32,768 and the same answer allowance. If a
stage has unresolved cap contact or excessive periodic-loop/answer-truncation rate, it is
setup-inconclusive; a loop-forced answer is never silently described as natural termination.

## Rationale and interpretation boundary

Extending these exact loops to 65k or 131k would spend tokens without adding reasoning, while
returning to 768 would censor every sample before the normal 2.3k–12.6k reasoning distribution
finishes. Periodic-loop termination occupies the principled middle: a generous envelope for all
productive reasoning plus an auditable escape from exact repetition. This changes inference
plumbing, not the macro hypothesis, tasks, prompts, sampling seeds, K values, or scientific
decision thresholds.
