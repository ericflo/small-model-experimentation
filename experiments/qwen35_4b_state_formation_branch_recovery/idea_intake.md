# Idea Intake: State-Formation Branch Authorization Recovery

## Decision

Proceed as a narrow operational recovery, not a scientific variant. The first authorized Stage-B
full-rank G0 stopped before model load because immutable producer v11 revalidates the LoRA-miss
receipt through the same nonlexical external-prefix helper that blocked analysis production.

## Closest work

The nearest experiment is `qwen35_4b_state_formation_analysis_recovery`. It repairs the exact path
only while producing one of five analysis receipts. It deliberately does not invoke model-bearing
producer stages, so it cannot make downstream authorization consumption work. Editing it would
change an already frozen, result-bearing recovery contract.

## Novelty and boundary

This successor wraps the unchanged producer CLI and source snapshot, installs the same exact-prefix
seam only for the duration of a branch-authorized producer invocation, and records recovery-owned
invocation provenance. It also preserves and retires the fail-closed G0 receipt pair that producer
v11 refuses to overwrite. It changes no model, data, metric, threshold, seed, optimizer, state
representation, scientific function, or branch rule.

## Decision-relevant controls

- Reproduce the original downstream branch-authorization rejection before installing the seam.
- Recompute the exact LoRA-miss authorization successfully through the seam without loading a model.
- Prove canonical equivalence for the registered prefix and clean descendants.
- Reject unrelated lexical aliases and traversal beneath the registered prefix.
- Restore the original function after normal and exceptional exits.
- Pin producer source/config/CLI/GPU-runner/analyzer, the analysis receipt, and the first recovery.
- Archive the exact canonical/mirror G0 failure before retiring either producer path.

## Expected disposition

A passing smoke licenses only a retry of already-authorized producer Stage B through this wrapper.
A model/G0/control/training failure after authorization is scientific or feasibility evidence under
the producer's existing taxonomy; a recovery-seam failure is mechanics evidence and stops execution.
