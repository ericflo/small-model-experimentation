# Idea intake: verified-macro capacity-fit vLLM rerun

## Program fit

- Primary program: `operator_and_skill_inventories`.
- Secondary connections: `test_time_reasoning_budget` and
  `structured_execution_and_compilers`.
- Existing or new program: existing. The scientific mechanism remains whether exact reusable
  composite operators improve bounded program search; this experiment repairs the inference
  protocol needed to reach that question.
- Related-work query: `make related QUERY="verified macro vLLM KV cache capacity max_num_seqs long context"`.
- Closest near-duplicate: `qwen35_4b_verified_macro_long_context_rerun`.

## Prior evidence and unresolved uncertainty

The original verified-macro experiment's 768-token thinking allowance and 128-token answer stage
were binding, so its failure did not falsify the macro mechanism. The direct long-context parent
raised those ceilings and reached the v2 induction smoke, but its high-rung diagnostics retained
`max_num_seqs=64`. A live cache around 995k tokens cannot conservatively hold 64 simultaneous
near-50k or near-63k contexts. Scheduler preemption/recomputation and extreme latency can therefore
masquerade as a reasoning-budget result.

The direct parent's output is not evidence for this follow-up: no output bytes are imported, pooled,
promoted, or scored. Its content-blind preflights contribute only frozen prompt identities and a
planning estimate of live cache capacity. The v2 smoke tasks have already been prompted in the
parent and therefore are not described as model-unseen; they remain unscored and fixed.

## Novelty claim

No completed experiment in this repository has run the frozen base/designed verified-macro smoke
under a scheduler configuration that (a) conservatively fits every active maximum-length sequence
inside the engine's live block-rounded KV capacity, (b) proves that fit after construction, and
(c) prevents termination probes or earlier scheduler protocols from becoming selection evidence.

The interesting research object is broader than one bug: long-context test-time compute has a
two-dimensional resource envelope. Raising token allowance without fitting concurrency to KV
capacity can produce a false semantic boundary. A reusable capacity-audited protocol could prevent
the same category error in later experiments.

## Mechanism

If scheduler overcommit is the remaining roadblock, reducing concurrency from 64 to a cache-fit
19 at 49k (or 15 at 61k) should let vLLM complete independent contexts without relying on an
unregistered preemption regime. A content-blind termination probe should then clear, and a newly
sampled matched-K base/designed matrix should become semantically eligible.

This explanation is weakened if the live capacity audit passes yet the registered probe or either
K=12 arm remains termination-inadequate at both rungs. That terminal outcome still localizes an
inference/setup boundary; it is not evidence that verified macros are useless because no eligible
semantic comparison exists. If termination clears but the designed ceiling fails the smoke gate,
the evidence instead points to an induction/interface ceiling under this frozen protocol.

## Controls

- Only `Qwen/Qwen3.5-4B` at the pinned revision, through the copied vLLM runner.
- Exact frozen tasks, libraries, demonstrations, prompts, sampling law, and semantic thresholds.
- Live cache capacity, cache block size, model context, active sequences, and worst rendered prompt
  are recorded and checked before generation.
- Fresh K=4 base probe at each rung; probe output is termination-only and structurally impossible to
  select as a K=12 arm.
- Fresh K=12 base and designed arms at the same selected rung and comparable runtime/engine identity.
- First adequate contiguous rung only; no row pooling, backfilling, or favorable cross-rung sample
  selection.
- Termination selection may use token identity only for the preregistered periodicity detector. It
  may not decode, parse, score, or inspect hidden examples.
- Full receipt and lower-rung termination re-verification before semantic access.

## Evidence output

- A live per-call capacity audit bound into each preflight and receipt.
- Exact raw vLLM bundles in a fresh external namespace with receipt-last commit semantics.
- A checksum catalog and content-blind first-adequate or terminal-unselected selection record.
- A post-gate base/designed semantic smoke analysis only if both K=12 arms are eligible.
- Learned operational guidance about jointly sizing context and scheduler concurrency, regardless of
  the semantic result.

## Decision

- Run: yes, after adversarial launch audit and GPU-owner coordination.
- New program: no.
- Expensive scope: at most one K=4 probe and two K=12 arms per rung, stopping at the first adequate
  matrix or terminal 61k rejection.
- Full capability evaluation: out of scope. Any positive smoke only licenses a separately frozen,
  matched-compute follow-up.
