# Idea Intake: State-Formation Branch Handoff Recovery

## Decision

Proceed as a narrow operational follow-up, not a scientific variant. The first frozen branch
recovery correctly archived and retired the pre-model failure and then produced a valid full-rank
G0. Its pathname-only retirement guard subsequently rejects that successful receipt because it
occupies the same canonical pathname.

## Closest work

The nearest experiment is `qwen35_4b_state_formation_branch_recovery`. Editing it would mutate a
frozen, result-bearing recovery after its smoke, archive, retirement, and G0 lineage were published.
Its exact-prefix seam and producer checks remain suitable; only downstream invocation orchestration
needs a byte/status-aware handoff.

## Novelty and boundary

This successor requires the exact archived failure, terminal retirement, successful G0 bytes and
identity, and first-recovery STARTED/COMPLETE receipts. It rejects a reappearing mirror, exact failed
bytes at the canonical slot, or any changed G0. It then calls the unchanged producer under the first
recovery's unchanged path seam and writes its own immutable invocation provenance. It changes no
model, substrate, training recipe, evaluation, metric, threshold, seed, or branch authorization.

## Expected disposition

A passing and published smoke licenses only already-authorized producer stages. A producer
G0/control/training/evaluation failure remains producer evidence; a lineage, seam, or handoff failure
is mechanics evidence and stops execution.
