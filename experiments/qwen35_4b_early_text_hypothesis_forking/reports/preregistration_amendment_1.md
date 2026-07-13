# Pre-model amendment 1: authenticated mechanics and stratified reachability

Frozen on 2026-07-13 after implementation-level adversarial review, before any
Qwen model construction, generation, outcome, or qualification-gold access.
The original design commit is `c064d7a4470a0638ddf4e483e82ccaa9f3720eed`.
This amendment withdraws that commit's underspecified Stage 1 contract and is
the new design boundary once committed and pushed. It closes ambiguities and
implementation bugs; it does not relax a threshold or promote the full-program
ceiling to causal evidence.

## Why an amendment is necessary

The original preregistration required exact mechanics execution, taskwise
deranged controls, and a separate full-program ceiling, but it did not specify
the full-program schedule. The first implementation happened to schedule four
parameter-free first operations. Passing those cases could not establish that a
bound integer parameter survives into a complete two-step proposal. Code review
also found that an aggregate adherence contrast could hide one failed context,
that padded controls had different terminal-token geometry, and that raw output
files were not sufficiently bound to their prepared requests and runner
receipts.

## Frozen mechanics clarification

The four diagnostic contexts still cross all 24 bound operations in each of
the systematic, independently deranged, duplicate, and placebo arms. In every
context separately, all of the following must pass:

- systematic supplied-operation execution at least `.75`;
- systematic candidate adherence at least `.60`;
- deranged supplied-operation execution at least `.60`; and
- systematic registered-target execution minus deranged registered-target
  execution at least `+.35`.

The registered aggregate support, duplicate/placebo, parse, and cap gates are
unchanged. The answer ABI is now enforced as exactly one terminal
`RESULT: [...]` line after the final `</think>` (apart from approved runner
terminal markers), matching the original natural-output requirement.

Duplicate and placebo injections are padded immediately before their preserved
terminal newline token. Each row therefore has identical injection length and
identical terminal-token geometry across all four arms. Derangements remain
independent across contexts and restricted within injection-length groups.

## Frozen full-program reachability ceiling

The ceiling has eight fresh public cases. Their registered first operations,
in order, are:

1. `reverse`
2. `running_sum`
3. `adjacent_diff`
4. `sort_desc`
5. `add_k(-2)`
6. `mul_k(3)`
7. `take_k(3)`
8. `rotate_k(2)`

Qualification requires all of:

- strict program parse at least `.90`;
- answer-limit contact at most `.05`;
- visible-pass rate at least `.50` over all eight cases; and
- visible-pass rate at least `.50` within the four parameterized cases.

This arm has no candidate-blind full-program comparator, so it is only a
non-causal reachability ceiling. It can show that the proposed interface is
usable on some complete programs; it cannot establish that the injected text
caused those programs, and it receives no causal or capability-gain credit.

## Frozen artifact and runtime boundary

Every prepared prompt is independently rebuilt from the pinned tokenizer,
config, public inputs, operation schedule, and control seeds before generation
or analysis. Every generated invocation uses a receipt-last transaction with
immutable `STARTED`, raw JSONL, runner metadata, and `COMPLETE` files. A valid
complete arm is skipped on resume. A `STARTED` arm without an authentic raw and
metadata pair is ambiguous and may never be regenerated; an authentic pair may
only be finalized without another model call.

Analysis authenticates exact row order and metadata, prompt token hashes,
decoded text against completion token IDs, natural-versus-forced continuation
structure, token accounting, deterministic seeds, runner schema and hash,
model/revision, exact engine and sampling arguments, the complete pinned Python
package set, and every aggregate count. Direct unauthenticated analysis is
forbidden.

The live engine must expose DP=TP=world-size 1, bf16, the exact CUDA-graph list,
the frozen scheduler, prefix caching off, Mamba cache mode `none`, and vLLM
`0.24.0+cu129`. Before generation, the largest prompt plus full registered
reserve is block-rounded. At the frozen maximum active width, required tokens
must fit the live KV-token capacity. This is intentionally a conservative
no-preemption gate, not merely an engine-admission check; failure stops the run
rather than silently reducing concurrency.

The exact critical-file map is verified both in the worktree and at the
implementation commit, and the design, amendment, implementation, and current
commits must all be published on `origin/main`. No model call is authorized
until the separate implementation-lock receipt is committed and pushed.
