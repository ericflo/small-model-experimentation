# Mechanics preflight incident

Status: attempt 1 aborted before the first experimental generation request;
append-only v2 repair is under adversarial review and is not yet authorized.

## What happened

The mechanics-only implementation and its separate lock were pushed and both
CI workflows passed. The exact pinned Qwen3.5-4B engine then initialized on the
RTX 6000 Ada, including vLLM's internal profiling and warmup. Before any
invocation transaction began, the live-preflight validator raised
`RuntimeError: live preflight cache geometry changed`.

The attempt-1 raw inventory contains only an empty advisory `run.lock` and
`raw/live_preflight.json` (SHA-256
`a438ecf190e4006e8f19368f907d37595a05d543ca8158c33d27333c816f14b5`).
There is no arm `STARTED` receipt, raw JSONL, runner metadata, completion
receipt, scored directory, authentication receipt, or summary. In the pushed
attempt-1 code, `_generate` writes `STARTED` before calling `runner.generate`,
and `_generate` is reachable only after live-preflight validation. Therefore
there were zero experimental generation requests and zero sampled model
outputs. This must not be described as zero model-forward work because engine
profiling and warmup occurred internally.

## Root cause

The recorded cache geometry was valid:

- 2,042 GPU cache blocks;
- 528-token attention blocks;
- 4,096-token Mamba blocks;
- three Mamba groups and one attention group;
- 11 blocks per maximum-length request;
- maximum concurrency `2042 / 11 = 185.63636363636363`; and
- token capacity `int(185.63636363636363 * 4096) = 760366`.

Pinned vLLM 0.24 intentionally floors the group-aware token-capacity field.
The validator incorrectly inverted that integer with a `1e-12` floating-point
equality. It also expressed per-arm fit in minimum-block token units, which is
not authoritative for this hybrid cache. Finally, it wrote a receipt already
labeled `LIVE_ENGINE_PREFLIGHT_PASS` before calling the validator. The embedded
PASS label in attempt 1 is therefore explicitly unauthenticated.

## Append-only recovery

Attempt-1 bytes remain immutable and hash-bound to implementation lock
`896c4cc64e157627eaf35a8a4365af766971644905fbde5c72e4d35ea72792e0`
and preoutcome receipt
`3de86e8b08bf37174cf687e4e7220ff802386c61652bafb6554cf1e772c89b88`.
The repair uses new `preoutcome_receipt_v2.json`, `implementation_lock_v2.json`,
`raw_v2/`, `scored_v2/`, and `summary_v2.json` paths. It authenticates the
vLLM floor, the exact frozen 11-block hybrid geometry, and the conservative
per-arm bound `active_sequences * 11 <= 2042`. It validates a candidate
preflight fully in memory before publishing a PASS receipt.

The v2 lock must disclose one prior engine initialization, zero experimental
requests, and zero sampled outputs; bind this report, the machine incident
receipt, the old lock, the old preflight, and both preoutcome receipts; and be
committed and pushed only after independent review and repository checks.

No benchmark, hidden, qualification, or confirmation content was read while
diagnosing or repairing this incident.
