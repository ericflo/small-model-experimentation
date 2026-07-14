# Adversarial mechanics failure postmortem

**Initial verdict:** `HOLD_REPORTING`

**Review scope:** post-generation classification and successor boundary

**Model/GPU access by reviewer:** none

**Raw sampled-output access by reviewer:** none

**Hidden/gold/benchmark access by reviewer:** none

An independent reviewer confirmed that `TERMINAL_INSTRUMENT_FAILURE` is the
only warranted mechanics classification. The 24/24 transport pass is narrow
interface evidence, while the 4,056 sampled outputs authorize no accuracy,
coverage, resource-match, oracle, or capability statement. Visible selection,
resource decision, and hidden result are absent, so hidden scoring remains
unauthorized.

## Root cause

The implementation conflated two temporal invariants:

1. initial transport authorization requires a complete transport transaction
   with every later invocation absent; and
2. historical replay must authenticate the immutable transport decision after
   a separately authenticated descendant chain has been appended.

`analyze_visible()` called `authenticate_transport_decision()`, which
recomputed `analyze_transport()` through the first invariant. The complete
`direct` transaction therefore caused deterministic rejection before the
full-chain verifier ran. The visible-analysis unit test mocked both critical
authentication functions and could not expose their composition.

## Required successor gates

- Distinct typed APIs for initial authorization and historical replay.
- An unmocked model-free lifecycle from transport decision through all four
  descendants and visible selection.
- Restart and crash recovery at every transaction boundary.
- Rejection of partial, gapped, reordered, foreign, and mutated descendants.
- A zero-generation-call proof for completed-inventory replay.
- A hash-bound automatic incident receipt on analysis failure.
- Fresh functions/tasks, request and record IDs, rendered prompt/token
  sequences, seed domains, ciphertext, and key.
- Collision receipts and an import sentinel forbidding use of this run's
  sampled bundles.
- A fresh adversarial implementation review, lock, and exact-commit green CI
  before any model request; committed-green visible selection before hidden
  access remains mandatory.

## Reporting hold resolution

The report's stale pre-execution claims that no result existed and model calls
remained zero were corrected as explicitly historical statements. Program
scorecards now route to the fresh-identity successor rather than the already
published lock and failed run. The current experiment is frozen as finished;
no claim ID is allocated and residual capability remains unadjudicated.

**Resolution state:** corrections complete; independent rereview pending.
