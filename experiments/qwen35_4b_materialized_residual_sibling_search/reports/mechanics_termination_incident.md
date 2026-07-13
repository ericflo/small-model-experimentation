# Mechanics termination-authentication incident

Status: attempt 2 is terminal after one 52-request mechanics invocation returned
in memory but before any sampled output was durably written. The invocation must
not be deleted, replayed, or described as a capability result.

## What happened

The append-only v2 repair and its separate lock were pushed, and both CI
workflows passed at execution head `6629de7f`. The exact pinned Qwen3.5-4B
engine initialized, the corrected live hybrid-cache preflight passed, and the
`suffix_materialized` transaction wrote its immutable `STARTED` receipt. The
runner then returned 52 rows and its metadata to `_generate`. Authentication
passed row counts, runner/runtime fields, sampling fields, and thinking-token
fields before raising:

`RuntimeError: runner termination authentication failed: suffix_materialized`

The frozen implementation authenticated the in-memory rows before writing the
raw JSONL and metadata. Consequently, attempt 2 contains an authenticated live
preflight and a terminal `STARTED` receipt but no raw JSONL, metadata,
`COMPLETE` receipt, score, authorization receipt, or summary. No later
invocation began. The model outputs were not printed or otherwise observed,
and the exited process left no recoverable output bytes.

## Root cause

The two termination IDs are deliberately different at the pinned revision:

- the text model configuration uses EOS token ID `248044`;
- the tokenizer uses `<|im_end|>` as its EOS token, ID `248046`; and
- the runner records the latter under `vllm_tokenizer_eos_ignored` because its
  sampling path explicitly ignores the tokenizer EOS and trims the model EOS.

The runner constructed the correct pair from the live pinned config and
tokenizer. Existing successful repository receipts also record
`248044`/`248046`. The experiment-local authenticator instead required
`248044` for both fields, while its unit fixture used a fake tokenizer whose
EOS was also `248044`. The test therefore encoded the same false assumption
and could not detect the live mismatch.

This was bookkeeping failure after generation, not evidence of model or
interface failure. It nevertheless destroys this attempt scientifically:
post-outcome code cannot recreate bytes that were never persisted, and the
immutable `STARTED` rule forbids resampling the invocation.

## Recovery boundary

Attempt-2 bytes remain immutable and hash-bound to v2 lock
`953da4e9ba5b4d19f5d1b785d907b7d78379705af50ab124a0027b7ce79a1264`.
The current experiment is sealed without a durable, authenticated model
result. Recovery must use a new experiment directory, new task/record
identities, and new sampling seeds; it may copy only the frozen scientific
design and corrected harness. Before any new model call, the successor must:

1. persist returned raw rows and metadata before semantic authentication as one
   durable generated bundle, or bind separate files with a single `GENERATED`
   hash receipt, then publish only a post-authentication `COMPLETE` receipt;
2. authenticate the exact live termination pair `248044`/`248046`;
3. test against the real pinned tokenizer metadata rather than only a fake
   tokenizer fixture;
4. bind this incident and both prior attempts into its preregistration and
   implementation lock; and
5. pass a new adversarial design and implementation review, commit/push/CI,
   and a separate clean lock publication.

No benchmark, hidden, qualification, or confirmation content was read while
running, diagnosing, or documenting this incident.
