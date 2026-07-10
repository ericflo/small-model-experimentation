# Preregistration amendment 4: evidence-based 16k budget and vLLM concurrency

Date: 2026-07-09. Frozen after interrupting the first 32k base-smoke batch, before it returned any
row and before rerunning the heldout interface or a complete scored smoke matrix.

## Observation

The 32k heldout interface decisively passed: 16/16 records covered, 64/64 valid macro-using samples,
nine proven periodic loops, zero unresolved cap contacts, and zero answer truncation. It established
that adequate thinking repairs the parent's apparent interface failure.

The subsequent base smoke remained in active decode for roughly 22 minutes without returning a
batch artifact. No generated output, parser result, visible score, oracle result, hidden label, or
task correctness was available or inspected. GPU saturation showed that vLLM serving overhead was
not the remaining bottleneck; output-token volume and the `max_num_seqs=32` scheduler ceiling were.

The completed train-only calibration already contains a safe lower operating point. At think@16,384,
55/64 samples closed naturally, all nine cap contacts pass the frozen periodic-loop detector, no
unresolved contact or answer truncation occurred, and natural p99 was 12,564 tokens. That is 76.7%
of the allowance: 3,820 tokens of headroom, missing the earlier 75% heuristic by only 276 tokens.
The earlier fraction was a conservative heuristic, not a model or scientific threshold.

## Frozen optimization

1. Set the productive-thinking headroom requirement to 80%.
2. Allow the already frozen periodic-loop detector from think@16,384 upward.
3. Use the registered ladder `[16384, 32768]`; 32k remains the automatic fallback.
4. Raise the throughput-only vLLM engine knob `max_num_seqs` from 32 to 64. The profiled cache has
   995,328 tokens, enough for roughly 57 concurrent solver sequences at prompt + 16k + answer
   reserve; vLLM may schedule fewer automatically when actual cache demand requires it.
5. Rerun the 4×16 train-only calibration at max-seqs 64 because batch composition can change
   sampled trajectories on Ada. Then rerun all 16×4 heldout interface records at the selected
   budget. No 32k interface row is carried into the new protocol.
6. Rerun the complete base/designed smoke matrix from scratch. The interrupted base call supplied
   no rows and cannot be pooled or scored.

All task data, prompts, model, revision, sampling temperature/top-p/top-k, seeds, K values, answer
allowance, loop detector, exact parser, interface threshold, smoke gates, and scientific decision
rules remain unchanged. Engine context remains 65,536, so prompt-plus-reserve headroom is ample.

## Inspection and freshness boundary

This amendment uses only train-only calibration termination metadata, the completed train-only
interface gate summary, GPU/cache telemetry, and wall time. Although the base smoke prompts began
generation, no output left vLLM before interruption and none was inspected. The fixed model has no
state carried between calls. The v2 tasks are therefore rerun without output-adaptive tuning; the
prompt and all task-level gates remain frozen.
