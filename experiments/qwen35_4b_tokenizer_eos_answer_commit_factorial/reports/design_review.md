# Adversarial Design Review

## Status

`HOLD_LIVE_CALLS`. Independent adversarial review of scaffold commit
`56b0b67b0b054610f783d6b6107a0e4c32b5c95e` returned `HOLD_DESIGN`.
The remediation is prospective and must receive a new independent review at an
exact pushed commit before construction or live calls.

## Self-review before delegation

- The experiment is a fresh result-bearing successor, not an in-place parser
  repair.
- Tokenizer EOS applies only to answer stages; thought semantics stay fixed.
- First-stop geometry and every pre-commit token are authenticated.
- HF EOS is a matched boundary control rather than historical prose.
- The two no-think prefix cells are paired and not called replications.
- Fresh task/function/request/seed identities are mandatory.
- Calibration remains known-answer interface measurement, not capability.
- Mechanics, protected labels, and benchmarks remain behind staged locks.
- The branch terminates if no tokenizer-EOS arm independently qualifies.

Independent review must inspect construction identity, boundary pairing,
malformed-stop controls, transaction order, matched-compute accounting, and the
hidden-label firewall before any model/GPU call.

## Independent review: first pass

The reviewer found four scientific blockers.

1. Boundary causality was not fail-closed. The phrase "matching sampled
   prefixes whenever both traces reach 248046" excluded precisely the early-HF
   and cap cases that could invalidate the comparison.
2. Thinking was not isolated across the four thinking cells. Matching numeric
   seeds cannot substitute for a single persisted shared-thought transaction
   because vLLM sampling is not batch-invariant.
3. The grammar froze two aliases while requiring both arity-two and
   arity-three blocks, and the unquoted YAML scalar `PROGRAM:` parsed as a
   mapping.
4. Conditional mechanics left task counts, strata, direct-pool bounds, resource
   matching, selector/inference, effect floors, terminal outcomes, and lock
   order adaptable after calibration.

The separate implementation review also noted that the scaffold runner is
still HF-EOS-only and that the smoke checker does not yet bind live stop/
finish/cap metadata, shared thought rows, transaction durability, or semantic
grammar. Those are expected scaffold limitations but independently block a
live lock.

## Prospective remediation

- All 192 tokenizer/HF pairs now fail closed on identical token prefixes
  through the earliest registered stop or cap, plus identical prompts, seeds,
  shared thought, injected prefix, adjacency, and batch geometry. Any mismatch
  terminates `BOUNDARY_PAIRING_INVALID` before cell metrics are used.
- Exactly one persisted thought row per task feeds all four thinking answers.
  Content from the first natural `</think>` onward is discarded, and every
  answer is rebuilt behind exactly one injected close.
- The grammar is arity-parametric for `k in {2,3}`, and `"PROGRAM:"` is quoted
  in configuration.
- Fresh seeds/namespaces, 24-task 8/8/4/4 strata, 24 candidates, 96 direct rows,
  24-row transport, two first-over resource matches, selector, inference,
  large-effect floors, terminal outcomes, and staged hidden-lock order are now
  frozen in the preregistration and config.

This document remains `HOLD_LIVE_CALLS` until the exact remediation commit is
pushed, both workflows are green, and the independent reviewer returns a
design pass. A design pass will not authorize model calls by itself; a later
implementation review and committed-green implementation lock are also
required.

## Independent review: second pass

Rereview of exact pushed/green commit
`3f5e6c10c927cac23a4198194b1b6f7c8ee35577` returned `HOLD_DESIGN`. It
confirmed that all four first-pass blockers were closed, then found five narrow
definitions that could still change qualification or interpretation:

1. dual boundary qualification within a matched thinking/prefix pair was
   listed as a usable outcome even though authenticated prefix equality makes
   the two >=44/48 exact-success sets mathematically incompatible;
2. parse success and cap contact were not independently defined;
3. calibration alias/stratum balance and a distinct transport namespace were
   absent;
4. visible correctness, cluster representative, arm-blind tie-breaking,
   abstention, and all-eight-hidden task success were underspecified; and
5. direct-pool exhaustion lacked a non-capability terminal, while a pass needed
   an explicit 24-task pilot scope.

These are now prospectively frozen in config and preregistration. Matched-pair
dual qualification is `SCORING_INVARIANT_VIOLATION`; parse is membership in the
registered arity token grammar; stop token 24 counts as cap contact;
calibration and transport balance/disjointness are explicit; selection uses an
arm-blind task/program hash and all-row exactness; exhaustion is
`DIRECT_RESOURCE_MATCH_POOL_EXHAUSTED`; and no pass is generalized beyond the
large-effect pilot. Live calls remain held pending another exact-commit design
rereview and a later implementation release review.

## Independent review: third pass

Rereview of exact pushed/green commit
`ef4e09ceb46a211d5bf8dcf92bb84863151108cf` returned `HOLD_DESIGN`. It
confirmed every second-pass repair, then found one mechanics ambiguity: suffix
arms emit two operations for a semantic candidate first operation, while direct
emits full arity-three programs, but downstream proposal identity was not
explicitly unified.

The prospective repair binds every parsed suffix as
`(semantic_candidate_first, suffix_1, suffix_2)` before any deduplication or
scoring. Direct parses directly to the same canonical three-operation tuple.
Every selector, execution, hash, hidden/oracle metric, and first-operation-
support count now operates on that full tuple and never on row index or sampled
order. Live calls remain held pending another exact-commit design rereview and
the separate implementation release review.

## Independent review: fourth pass

Rereview of exact pushed/green commit
`a7c183f3c2f9b47a0913b77c998b09eefd4fb9c1` returned `HOLD_DESIGN`. It
confirmed the canonical full-triple repair end to end, then found that the
termination controls conflated a well-formed but grammatically early stop, a
valid exact-cap length event, and malformed stop geometry.

The prospective repair freezes three classes. A unique final stop with matching
reasons is authenticated and its preceding content is scored normally. An
exact-24 length event without the registered stop is authenticated, counts as a
cap contact, retains all 24 IDs as content, and is scored normally. Claimed-
stop-without-stop, short length, repeated/interior/post-stop tokens, or wrong
reasons are authentication failures. Live calls remain held pending another
exact-commit design rereview and the implementation release review.

## Independent review: fifth pass

Rereview of exact pushed/green commit
`2fb46a03c273161d09480300123f2ae1fb9aaa52` returned `HOLD_DESIGN`. It
confirmed terminal classification, then found two global contradictions. The
selector referenced an undefined normalized tokenizer encoding even though
prefix inventories may tokenize differently, and the exact-success disjointness
proof ignored shared exact-cap length traces.

The prospective repair gives every full semantic operation tuple an injective,
arm/prefix/tokenizer-independent base-24 ID in 0-13,823 and serializes it as
exactly two-byte big-endian for hashing. It also replaces disjointness with the
cap-bounded-overlap proof: global overlap <=2 makes `44+44-2>48`, and per-arity
overlap <=1 makes `22+22-1>24`. Live calls remain held pending another exact-
commit design rereview and implementation release review.

## Independent review: sixth pass

Rereview of exact pushed/green commit
`d919a969fd8bd0d3b5bfdf170421fda77530716a` returned `HOLD_DESIGN`. The
focused audit passed the semantic program ID and overlap proof, but the global
scan found stale summaries in the intake and README that still attributed
dual-qualification impossibility to prefix equality plus exactness alone.

Those summaries now explicitly include the cap ceilings and the same bounded-
overlap arithmetic as the preregistration: overlap <=2 gives `44+44-2>48`, and
overlap <=1 per arity gives `22+22-1>24`. Live calls remain held pending another
exact-commit design rereview and implementation release review.
