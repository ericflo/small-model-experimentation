# Idea Intake: Commit-Slot Jacobian Value Transport

## Program Fit

- Programs: `interpretability_and_diagnostics`, `test_time_reasoning_budget`,
  and `structured_execution_and_compilers` for the fixed latent interface.
- Closest duplicate: `qwen35_4b_forced_commit_jacobian_value_transport`.
- Other anchors: C40 single-token implicit confidence,
  `qwen35_4b_jacobian_transport_control_replication`, and C51.

## Prior Evidence

1. Exact-control J coordinates transport a context-local concept through a later
   computation, but remain oracle.
2. Native thought never closed on 48/48 through 1,024.
3. Close-only forced output parsed only 12.5%--18.8%, usually restarted analysis,
   and opened no J stage.
4. C40 shows useful self-knowledge can live in a concentrated answer-token
   distribution when verbalized reporting fails.

## Novelty Claim

No prior experiment supplies only fixed answer syntax, measures a constrained
semantic alias choice against no-thought, exact-length shuffled-thought, and
free-form controls, then asks whether task-general correctness probability is
causal in replicated J space.

## Mechanism

The slot removes emission-format entropy while preserving semantic uncertainty.
If thought content has resolved the task, the constrained alias distribution
should beat both an immediate no-thought slot and a permutation of the identical
thought-token multiset, while varying across traces. A scalar J value axis must
then explain correctness beyond direct alias logits and generic margin.

## Control Plan

- Baselines: no-thought slot; exact-length shuffled-thought slot; same-prefix
  close-only free-form; unconstrained alias mass and top token.
- Mechanism controls after value: correct-alias J activity, slot margin,
  shuffled beta, two exact random arms, logit lens, identity/full donor, raw,
  ActAdd, and matched non-J.
- Fresh splits/seeds at every stage; no benchmark content.
- Gold correctness is oracle and never available to a deployed controller.

## Evidence Output

- Preserve every gate result and update program/synthesis strategy.
- No claim ID during re-grade.
- Branch only replicated scalar causality into a non-oracle matched-compute test.

## Decision

Run as a new staged experiment. CPU smoke caught and repaired a pre-model seed
collision; the final 96 rows are fresh and the adversarial review is frozen.
