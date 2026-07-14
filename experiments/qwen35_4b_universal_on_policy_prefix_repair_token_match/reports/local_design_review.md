# Adversarial Review: Fresh Local Gate and Deployment Protocol

## Scope and observation boundary

This review covers the fresh local capability substrate, trained-arm deployment,
absolute and control-relative promotion rules, and checkpoint order after paired
training. It was completed before any local model call, trained-arm merge, local
completion, capability score, or benchmark access. It inspected only
experiment-owned procedural code, published training/parent receipts, external
adapter identities, and the repository's inference contract. It read no
`benchmarks/` content.

## Fresh task and hidden-label audit

- Seed 88,009 deterministically freezes 26 truth-audited tasks, exactly two for each
  of the 13 registered universal skills. Source/model-input hashes are
  `9682744e...acdee` / `ff407551...ce988` and design-receipt hash is
  `3982d5b8...6e85a`.
- Every task id is namespaced `local88009_*`. All 26 model-facing rows contain only
  `id`, `messages`, and public `meta`; answer, target trace, and executable audit are
  absent.
- Canonical message bytes were compared with 658 messages from the two frozen
  training streams and parent-collection source, yielding zero overlap. The same
  generator and mix were replayed at reserved prior local seeds 88,000–88,008,
  covering 234 messages, again with zero overlap.
- The local source is materialized only after both adapters have trained. It cannot
  affect selection, stream construction, optimizer settings, or adapter weights.

## Pre-outcome backend amendment

The original preregistration named a single Transformers process for the local
comparison. The current repository-wide inference contract now requires the pinned
vLLM template for bulk generation and prohibits backend mixing unless Transformers
internals are themselves the measurement. This gate measures behavior, not
Transformers internals. Retaining the old backend would violate the active operating
guide.

The amendment is therefore frozen before any local model call or score: parent,
replay control, and prefix-repair candidate all run through the same experiment-local
vLLM runner, natural-thinking channel, greedy decoding, seed 88,009, 1,024-token cap,
4,096 context, 16-sequence scheduler, 8,192 batched-token limit, and explicit CUDA
graph sizes 1/2/4/8/16. The change is symmetric across arms and creates no
post-outcome choice.

Runtime vLLM LoRA is a verified silent no-op for these Qwen3.5 PEFT adapters.
Consequently, the published parent composite is reused and each trained arm must be
explicitly merged into the same composite architecture. The merger authenticates
the committed training receipt and external adapter, requires every applied LoRA
module to be nonzero, and hashes every full weight file. Passing adapters directly
to vLLM is forbidden.

## Checkpoint and transaction audit

The only authorized order is:

1. publish this local design;
2. merge the replay control, preserve and publish its receipt;
3. merge the prefix-repair candidate, preserve and publish its receipt;
4. run one local stage containing all 26 prompts for each of the three composites;
5. preserve the raw output, metadata, log, consolidated grading receipt, and an
   explicit promotion or empty-promotion receipt.

Each step requires a clean worktree and its prerequisite receipts committed
byte-for-byte at `HEAD`, followed by smoke, `make check`, fetch/rebase, both gates
again, push to `main`, and both GitHub workflows. A partial local event is preserved
and never silently deleted or resampled. Raw runner metadata must bind model path,
runner hash, input hash, sampling, engine geometry, row counts, and preflight commit.

## Gate audit

The absolute candidate gate is unchanged in substance from the predecessor and is
integerized for 26 rows: at least 24 parses, 17 correct, at most two cap contacts, at
most one feasible-route abstention, and at least one of two correct separately for
execute, induct, and probe. Candidate promotion additionally requires strict total
correct wins over both the unchanged parent and exact-compute replay continuation,
plus strict wins over both on the six execute/induct/probe rows. A tie fails. Parent
and replay absolute gates are recorded for diagnosis but cannot substitute for a
candidate absolute pass.

The evaluator parses only the last exact `ANSWER:` line, grades exact procedural
truth, records cap contact from the raw vLLM termination fields and sampled-token
count, and keeps raw token-bearing outputs for audit. No train loss, merge norm, or
control weakness can trigger promotion.

## Remaining risks

1. Two items per skill make the gate noisy. The fixed strict controls reduce false
   promotion but can reject a useful adapter; thresholds cannot change after output.
2. The candidate has 33,421 fewer target/nonzero-weight tokens and 6,261.8 less
   absolute loss mass than replay despite equal forward tokens. A win supports the
   package under matched forward compute, not isolation of prefix conditioning.
3. Three sequential vLLM model loads are one logical local stage but not one engine
   process. Every arm has identical runner bytes and batch geometry; no claim of
   common-random-number pairing is made.
4. The gate reuses a longstanding procedural generator. Fresh seeds, namespaced ids,
   and byte-level checks prevent item reuse, but a positive still concerns transfer
   within this broad procedural interface before any held-out benchmark claim.

No issue permits outcome-aware tuning or contaminating access. The task source,
backend, merge order, and promotion rule are frozen and runnable.

**Verdict:** `PASS_CONTROL_MERGE`.

This verdict authorizes only `merge-control` after this design is committed, rebased,
pushed, and green in both workflows. Candidate merge requires the separately
published control-merge receipt. Local generation requires both published trained-arm
merge receipts. Benchmark and aggregate seed 78,139 remain sealed.
