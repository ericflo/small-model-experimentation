# Preregistration

## Status and scope

Prospective and model-free as of 2026-07-14. No task construction, calibration
sample, mechanics sample, or protected label has been opened for this
experiment. The first independent design review returned `HOLD_DESIGN`; this
revision freezes its causal-pairing, shared-thought, grammar, and conditional-
mechanics corrections. Any amendment after model output must be a new
experiment.

## Scientific question

Does first tokenizer EOS provide a strict answer-stage commit boundary on fresh
known-answer rows, and—conditional on independent qualification—does the
resulting interface expose a materialized-residual capability gain?

## Fixed model/backend

- `Qwen/Qwen3.5-4B`
- revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- bf16, vLLM 0.24.0+cu129, identical engine and runner within every comparison
- no adapter during interface or mechanics evaluation

No other model may generate, judge, label, or teach.

## Fresh construction and collision receipts

Construction seed `2026140700` creates exactly 48 calibration and 24 mechanics
tasks under namespace `tokenizer-eos-answer-commit-factorial-v1`. Each task is
an exact-depth-three procedural list transform with eight visible examples,
eight sealed hidden examples, and sixteen unlabeled probe inputs. Every public
instance, target function, target triple, target suffix, and input row is
unique across splits. No visible relation has a depth-zero/one/two solution.
Publicly live first operations are exhaustively labeled over all 576 two-
operation suffixes.

The 24 mechanics tasks have frozen 8/8/4/4 strata with exactly 1/2/3/4
publicly live first operations. Before any call, the constructor must publish
hash-bound receipts proving zero overlap with the predecessor and repository
catalog in task IDs, function fingerprints, request IDs, rendered prompts,
seed keys, derived seeds, and rendered prompt-token sequences. Calibration and
mechanics records, direct-pool order, shuffle maps, and hidden ciphertext/hash
are frozen in that receipt. Calibration code is forbidden from reading any
mechanics or hidden row. `benchmarks/` is never read.

Each 24-row calibration arity block independently has the same 8/8/4/4
mechanics-representative live-first-operation strata. Within each arity block,
every A-X alias occurs exactly once in every answer position: two independently
balanced position permutations for arity two and three independently balanced
position permutations for arity three. The transport request namespace is
separately frozen as `tokenizer-eos-answer-commit-factorial-transport-v1` with
seed `2026140706`. The collision receipt must additionally prove that every
transport request ID, seed key, derived seed, and prompt-token sequence is
disjoint from calibration and bulk mechanics.

## Arity-parametric token-native grammar

Calibration contains 24 arity-two and 24 arity-three mechanics-length echo
rows. The registered output is tokenized once and is exactly
`PROGRAM: <alias_1> | ... | <alias_k>`, where `k` is the row's registered
output arity in `{2, 3}` and every alias is one of the frozen A-X operation
aliases. In the freeform cells the entire registered string is sampled. In the
program-slot cells the literal token IDs for `PROGRAM:` are injected and the
remaining registered token sequence is sampled. Exactness compares injected
plus sampled pre-commit token IDs with the registered token IDs. The commit
token is boundary metadata. No decoded-text trimming, re-tokenization, grammar
mask, logit bias, or teacher-forced answer identity is allowed.

For each prefix condition, the tokenizer receipt separately enumerates and
hashes every grammatical token-ID sequence for A-X programs at each registered
arity (24^2 plus 24^3 strings). The freeform inventory encodes the whole
string; the slot inventory composes the separately registered injected prefix
IDs with the registered sampled-remainder IDs. The receipt proves both
inventories render the same grammatical bytes but does not assume their token
segmentation matches. Parse success means that condition's composed pre-commit
IDs are a member of its row-arity set, regardless of whether the program is the
known answer. Exact echo success separately requires equality with that
condition's one registered known-answer sequence. Thus parse and exact are
distinct frozen metrics and neither depends on decoded text.

## Calibration factorial

Cross:

1. answer stop boundary: first tokenizer EOS 248046 versus HF model EOS 248044;
2. thought policy: no-think versus one persisted 512-token thought prefix; and
3. answer prefix: freeform versus literal `PROGRAM:` token prefix.

This is eight 48-row cells, 384 answer requests, and 192 registered boundary
pairs. All use `n=1`, temperature 0.6, top-p 0.95, top-k 20, and a 24-sampled-
token answer cap. Requests in each tokenizer/HF pair are adjacent in the same
vLLM invocation with identical prompt token IDs, task/record identity, numeric
answer seed, sampling policy, row order, scheduler mode, and batch geometry;
only `stop_token_ids` differs. Prefix conditions deliberately have different
answer prompts and are paired conditions, not replications.

### One shared thought transaction

For each calibration task, exactly one thought transaction is generated from
the prefix-free base prompt using the HF-EOS thought policy and seed domain
`thought`, with a 512-token cap. Its full sampled IDs, finish/stop state, and
retained IDs are durably persisted before any thinking answer continuation.
If `</think>` occurs, retain only IDs before its first occurrence and discard
the close plus every naturally generated answer token after it. If it does not
occur, retain all nonterminal thought IDs. Trim only a terminal registered HF
EOS from this thought transaction. Every one of the four thinking answer cells
then rebuilds a new answer stage from the exact same persisted retained IDs,
followed by exactly one injected `</think>\n\n` and that cell's optional answer
prefix. Thus natural close never bypasses the registered answer stage and the
tokenizer-EOS policy is never applied to thought generation.

### Fail-closed all-pair boundary authentication

Every one of the 192 boundary pairs must authenticate, including pairs that do
not emit 248046. For each pair, compare sampled token IDs from position zero
through and including the earliest authenticated terminal event in either
trace: its registered stop token, or its 24-token cap. The prefixes must be
identical. Prompt IDs/hashes, answer seed, sampling parameters, shared-thought
receipt/hash when applicable, injected prefix IDs, request adjacency, engine
metadata, row order, and batch geometry must also be identical. A length event
must contain exactly 24 sampled answer IDs and may not masquerade as a stop; a
stop event must contain its registered stop exactly once at its final sampled
position and have matching finish/stop reasons. Any divergence, missing pair,
mutation, ambiguous first stop, or unmatched event terminates the entire
experiment as `BOUNDARY_PAIRING_INVALID` before qualification or mechanics.

An answer-cap contact is `len(answer_stage_sampled_token_ids) >= 24` or
`finish_reason == "length"`. Sampled IDs include the registered terminal stop,
so a valid stop emitted as sampled token 24 is still a cap contact. A stop at
or before 24 can be authenticated and exact while independently counting
against the cap-contact gate.

Within each matched thinking/prefix pair, the only valid qualifier outcomes are
tokenizer-only, HF-only, and neither. Under authenticated paired prefixes,
tokenizer-exact and HF-exact are disjoint row sets: the first registered
terminal ID for one policy is strict content for the other. Two disjoint sets
cannot both meet 44/48. Observing both cells qualify in the same matched pair is
therefore `SCORING_INVARIANT_VIOLATION`, not a scientific outcome and never
mechanics authorization. Qualifiers in other HF thinking/prefix cells are
reported but do not replace the HF control paired to the selected tokenizer
cell. If no tokenizer cell qualifies, any HF qualifier yields the HF-only
terminal; no qualifier yields neither. Only tokenizer-only in the selected
matched pair supports a causal termination-boundary claim.

## Interface gates and selection

A cell qualifies only if all conditions hold:

- at least 44/48 strict parses and 44/48 exact echoes;
- no more than 2/48 answer-cap contacts;
- in each 24-row arity block, at least 22 parses and 22 exact echoes and no
  more than one answer-cap contact; and
- zero transaction, token, stop, prompt, seed, thought, cost, or inventory
  authentication failures.

After the all-pair gate passes, select the first qualifying tokenizer-EOS cell
in this least-departure order:

1. no-think, `PROGRAM:` slot;
2. no-think, freeform;
3. shared think@512, `PROGRAM:` slot;
4. shared think@512, freeform.

Observed metric ranking cannot change the order. HF cells are controls and
cannot be selected. If no tokenizer cell qualifies, publish either
`HF_ONLY_CONTROL_QUALIFIES_TOKENIZER_FAIL` or
`NO_VALID_TOKENIZER_EOS_ANSWER_SEAM`, open no mechanics file, and retire the
branch.

## Required termination and transaction controls

Before a live lock, model-free and malformed-runner tests must reject missing,
early, repeated, interior-plus-terminal, and post-stop registered tokens;
wrong stop or finish reason; cap overflow or short output relabeled length;
extra newline, close, chat marker, or byte before commit; tokenizer stopping on
thought; natural-answer bypass; pair-prefix divergence; pair prompt/seed/
thought mutation; and prompt/token/text/cost/summary mutations. Append-only
transactions must crash safely and authenticate exact stop configuration,
first-stop sampled IDs, pre-commit IDs, physical and logical costs, runner,
backend, model, and revision.

## Staged publication and lock order

The immutable order is:

1. construct and hash all fresh public/calibration/mechanics/direct/shuffle
   inventories without opening hidden plaintext;
2. commit and push the preoutcome receipt and obtain green CI;
3. commit and push a calibration implementation lock binding every executable,
   request byte, config, tokenizer receipt, environment receipt, and test; then
   obtain green CI;
4. make the first calibration call, publish complete authenticated
   transactions and the terminal/selection receipt, commit/push, and obtain
   green CI;
5. if and only if a tokenizer cell qualifies, publish a distinct mechanics
   lock binding the selected interface and sealed mechanics bytes, then obtain
   green CI;
6. run transport and, if it passes, bulk generation in the frozen order below;
7. publish and commit the complete visible-only selector plus both taskwise
   direct-prefix resource-match plans, push, and obtain green CI; and
8. only then open hidden plaintext to score already frozen selected IDs and
   oracle coverage.

No failed gate can be repaired in place or skipped.

## Frozen selected-interface transport

Transport uses 24 disjoint public-visible known-answer echoes in alternating
order: 12 arity-two suffix-shaped and 12 arity-three direct-shaped rows. The
selected boundary, thinking policy, prefix, seeds, caps, prompt construction,
backend, and runner remain unchanged. Transport requires at least 22/24 exact
echoes and parses, at most one cap contact, and at least 11/12 exact echoes and
parses in each arity. Failure is `SELECTED_INTERFACE_DID_NOT_TRANSPORT` and no
bulk mechanics runs.

Transport IDs and seeds derive only from the frozen transport namespace/seed;
they must pass the preoutcome disjointness receipt against calibration and
every bulk arm before the calibration lock can be minted.

## Frozen conditional mechanics

Mechanics uses the 24 fresh tasks and, per task:

- 24 materialized candidate-state-to-target relations, one for every first
  operation;
- 24 name-only original visible relations in the same candidate order;
- 24 target-deranged materialized relations from seed `2026140705` and frozen
  task-hash permutation; and
- one candidate-blind direct master pool of exactly 96 rows from seed
  `2026140703`.

Suffix arms share request IDs, candidate order, answer/thought seed derivation,
caps, and batch geometry. Invocation order is transport, complete direct master
pool, materialized, name-only, shuffled. The direct pool cannot be extended.
If either resource target exceeds the complete 96-row direct pool, stop before
visible selection or hidden access as `DIRECT_RESOURCE_MATCH_POOL_EXHAUSTED`.
This is a non-capability resource-design failure, not a materialized-residual
failure.

For each task, the complete 24-row materialized arm freezes two conservative
first-over prefixes of the already generated direct pool. Sampled-token cost is
the count of every stage-one and stage-two sampled token ID, including terminal
IDs. Logical-model-token cost is every actual stage-one/stage-two prompt token
processed plus every sampled token; forced close and slot-prefix IDs count in
their actual continuation prompt positions. For each metric, select the
shortest frozen-order direct prefix whose cumulative cost is at least the
materialized cost. Record target, achieved cost, overshoot, row IDs, and pool
exhaustion. No outcome may affect a prefix.

## Frozen selector, estimands, and inference

The deployable primary is task-level hidden accuracy of
`visible-probe-consensus-v1`, selected before hidden plaintext opens. For each
arm/task it rejects programs that fail to parse, execute, or exactly match the
public target on every one of the eight visible rows. It deduplicates canonical
token-ID programs and clusters survivors by their output vector on the 16
unlabeled probes. It chooses the largest cluster; tied clusters are ordered by
the minimum member hash of the canonical byte string UTF-8
`visible-probe-consensus-v1`, NUL, UTF-8 task ID, NUL, then each program token
ID as unsigned four-byte big-endian. The member with minimum same hash
represents the winning cluster; an exact hash collision falls back to
lexicographically smallest token-ID tuple. This tie-break contains no arm,
candidate index, sampled order, hidden value, or outcome. If no survivor
exists, the selector abstains and the task scores zero. The same selector
applies to materialized, name-only, shuffled, and both direct prefixes.

A selected program succeeds on a task only if it executes and exactly matches
the target on all eight hidden rows. Oracle coverage succeeds only if at least
one proposal in that arm/task does so. Partial hidden-row correctness never
counts as task success.

Oracle proposal coverage is diagnostic. The diagnostic exhaustive CPU ceiling
enumerates all 13,824 frozen depth-three programs, filters on public-visible
rows only, and later reports whether any survivor is hidden-correct. Paired
task-level selected and oracle contrasts receive deterministic 10,000-resample
paired-bootstrap 95% intervals and one-sided exact McNemar tests using seed
`2026140704` plus frozen arm offsets. Inference and the ceiling are report-only
and cannot alter the decision.

`TOKENIZER_EOS_MATERIALIZED_RESIDUAL_LARGE_EFFECT_PILOT_PASS` requires all:

- parse rate >=0.90 and cap-contact rate <=0.05 in every generation arm;
- materialized selected hidden accuracy >=0.25 and at least 6/24 successes;
- selected-accuracy gains >=0.125 versus name-only, shuffled, sampled-token
  matched direct, and logical-token matched direct;
- materialized oracle coverage >=0.35;
- oracle-coverage gains >=0.125 versus those same four controls; and
- at least eight distinct first-operation aliases among hidden-correct
  materialized proposals.

If an authenticated generation ABI gate fails after transport, emit
`MECHANICS_INTERFACE_NONTRANSPORT`. Otherwise failure is
`TOKENIZER_EOS_MATERIALIZED_RESIDUAL_LARGE_EFFECT_PILOT_FAIL`.

Even a pass is evidence only for a contamination-free 24-task large-effect
pilot under this frozen substrate and selector. It is neither confirmatory
evidence nor a general claim that the interface installs a deployable
capability.

## Non-rescue rules

No task count, stratum, prompt, arm, priority, parser, cap, temperature,
threshold, seed, direct-pool ceiling/order, invocation order, selector,
backend, resource metric, or terminal label may change after calibration
begins. No cap increase, parser relaxation, suffix trimming, alternate
extraction, extra direct sample, threshold tuning, cheap-ranker revival, or
outcome-conditioned rerun is allowed. Negative results and transaction
incidents remain published.
