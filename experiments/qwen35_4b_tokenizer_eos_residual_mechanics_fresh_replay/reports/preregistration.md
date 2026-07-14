# Preregistration: Tokenizer-EOS residual mechanics fresh replay

**State:** draft frozen for independent adversarial design review; no model call
is authorized

**Date:** 2026-07-14

## Parent incident and recovery boundary

The scientific parent
`qwen35_4b_tokenizer_eos_answer_commit_factorial` qualified the no-think
tokenizer-EOS `PROGRAM:` interface on fresh calibration (48/48 strict exact and
parse in both prefix cells, versus 0/48 for every HF-model-EOS control) and
passed mechanics transport 24/24. It then generated 4,056 durable outputs but
failed before visible selection because historical transport replay reused the
initial invariant that later invocations must be absent.

This successor preserves that failure as terminal. It must not import, inspect,
rescore, or derive any task, prompt, threshold, selector, or control from the
parent's raw sampled bundles. It may authenticate an exact allowlist of parent
administrative evidence: preregistration, design/review reports, configuration,
source, construction receipts, calibration decision, implementation locks,
aggregate transport decision, and terminal failure receipt. Any path under the
parent's `runs/mechanics/raw/` is forbidden.

## Question and frozen scientific design

On genuinely fresh exact-depth-three tasks, can materializing every candidate
first operation's public consequences let Qwen3.5-4B generate useful
two-operation residuals and beat:

1. all-24 name-only candidate relations;
2. all-24 token-preserving semantically shuffled relations;
3. candidate-blind full-program sampling at the taskwise sampled-token
   first-over budget; and
4. candidate-blind full-program sampling at the taskwise logical-model-token
   first-over budget?

The arms, prompt ABI, no-think tokenizer-EOS answer boundary, selector,
matched-compute definitions, metrics, thresholds, and stop rules are copied
without outcome-dependent change. The only scientific-parent change is fresh
data and identity. The only implementation change is separating initial
transport authorization from historical immutable replay.

## Model and backend

- Only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- bf16 on the pinned repository vLLM 0.24.0+cu129 stack for every arm.
- No other model, teacher, adapter, training, benchmark, benchmark-derived
  content, or backend mixing.
- Tokenizer EOS `<|im_end|>` ID 248046 is the answer-stage stop. HF model EOS
  ID 248044 remains the matched boundary control during fresh calibration.

## Fresh identity and contamination controls

- New namespace: `tokenizer-eos-residual-mechanics-fresh-replay-v1`.
- New seed block: `2026140800`--`2026140806`.
- Regenerate all calibration and mechanics tasks. Reject every parent function
  fingerprint before split assignment, not merely concrete task rows.
- Require zero overlap with the parent in task IDs, record IDs, request IDs,
  canonical seed keys, derived sampling seeds, identity-free rendered prompts,
  and rendered prompt-token sequences.
- Preserve the A--X operation semantics and grammar; freshness comes from
  functions, examples, prompts, identities, and seeds rather than changing the
  scientific language.
- Generate a new mechanics-gold ciphertext and new ignored key. Neither parent
  ciphertext nor parent key may be copied.
- A preoutcome collision receipt must bind every checked parent inventory and
  fail closed on missing, symlinked, malformed, extra, or hash-drifted files.

## Temporal transaction repair

Two typed public APIs are mandatory:

1. `authorize_initial_transport` accepts only the exact state in which
   transport is complete and every later invocation is absent. It computes and
   durably writes the transport decision once.
2. `authenticate_historical_transport` accepts only an exact immutable
   transport transaction and decision plus a separately authenticated complete
   descendant chain. It permits valid descendants but never recomputes the
   initial later-absent authorization.

Visible analysis must authenticate the complete five-invocation chain before
historical transport replay. Descendant registrations remain bound to the
exact transport-decision hash. Partial, gapped, reordered, foreign, duplicated,
or mutated descendants fail closed. Completed-inventory replay must make zero
generation calls.

Every invocation remains:

`STARTED -> generate -> durable GENERATED + receipt -> authenticate -> COMPLETE`.

Restart may authenticate an intact durable bundle but may never resample after
`STARTED`. Analysis failure must create an automatic exclusive, hash-bound
incident receipt without modifying generation artifacts.

## Mandatory model-free lifecycle tests

- One unmocked end-to-end synthetic lifecycle: create transport, authorize and
  store its decision, append all four descendants, authenticate the complete
  chain, historically replay transport, and produce visible selection.
- Every valid contiguous inventory prefix and the complete chain.
- Every partial successor, gap, reorder, foreign registration, Boolean/integer
  type alias, content mutation, predecessor mutation, callback mutation, and
  publication-time mutation.
- Crash/restart after each durable boundary, with zero recovery resampling.
- `--stage run` restart from a complete five-transaction inventory.
- Zero model/generation calls during post-chain replay and visible analysis.
- Parent sampled-bundle import sentinel and full path-audit tests.
- Hidden authorization denial until the visible receipt is committed, pushed
  to `main`, and green at the exact commit.

## Fresh calibration and mechanics

Fresh calibration repeats the frozen 2 x 2 x 2 factorial over tokenizer versus
HF EOS, no-think versus think512, and freeform versus `PROGRAM:`. Only a
tokenizer-EOS cell with at least 44/48 exact, 44/48 parse, at most 2/48 caps,
and the corresponding 22/24, 22/24, at-most-1 per-arity gates may advance. The
priority order remains structured no-think, freeform no-think, structured
think512, freeform think512. Any result other than the same qualified interface
stops mechanics.

Fresh transport has 24 rows balanced 12/12 across suffix/direct arities. It
requires at least 22/24 exact and parse, at most one cap, and at least 11/12
exact and parse per arity. Failure stops all later mechanics.

Mechanics uses 24 fresh tasks, all 24 candidates per suffix arm, a 96-sample
direct master pool per task, eight visible examples, eight hidden examples, and
16 unlabeled selector probes. Proposals deduplicate by canonical full
three-operation tuple. The selector uses visible exactness and unlabeled-probe
consensus only, with an arm-blind hash tie-break.

## Estimands and decision

Primary deployable task success is exact correctness on all eight hidden
examples, with abstentions scored zero. The materialized arm must achieve at
least 0.25 selected accuracy and six successful tasks, and improve by at least
0.125 over name-only, shuffled, sampled-token-matched direct, and logical-token-
matched direct. Its oracle proposal coverage must reach 0.35 and exceed each
same comparator by at least 0.125, with support from at least eight first
operations. All generation arms must reach at least 0.90 parse and at most 0.05
cap contact. Exact paired tests and 10,000-resample task bootstrap intervals are
reported as pilot evidence; no confirmatory generality claim is allowed.

Possible terminal outcomes are interface invalid, transport failure, resource
pool exhaustion, mechanics ABI failure, large-effect pilot pass, large-effect
pilot fail, or terminal instrument failure. No threshold may be relaxed after
observing any output.

## Release sequence and hidden boundary

1. Commit and publish this design; obtain independent `PASS_DESIGN`.
2. Construct fresh data model-free and publish collision receipts.
3. Implement and independently review the exact code commit.
4. Publish a model-free implementation lock and require exact-commit green CI.
5. Run fresh calibration; publish its decision and require green CI.
6. Publish a winner-bound mechanics lock and require green CI.
7. Run transport and mechanics, then visible analysis.
8. Commit visible selection, push `main`, and require both workflows green.
9. Only then may the fresh key be opened and hidden scoring run once.

Any code or data change after a lock invalidates that lock. Benchmark contents
remain unread throughout.
