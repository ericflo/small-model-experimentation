# Backlog

## Next Experiments

- LoRA architecture counterfactual completed at its registered pilot stop:
  `qwen35_4b_state_carry_vs_state_bag` emitted valid `PILOT_MECHANISM_MISS`. The fixed-source pilot
  was complete and matched, but Carry joint state accuracy was 0.00459 versus the 0.40 gate; the
  +0.043 answer effect was uncertain and swaps were noncausal. Do not run its confirmation or
  sample-more stages, and do not reinterpret the earlier invalidated analysis-dispatch attempt.
- **Next mandatory experiment:** execute the now-created and adversarially reviewed
  `qwen35_4b_state_carry_vs_state_bag_fullrank_delta` successor specified by the LoRA
  preregistration. It replaces rank-32 LoRA with zero-initialized full-rank weight deltas on
  Qwen layers 12–19, enabled only during extra R applications. Keep `Qwen/Qwen3.5-4B`, the frozen
  base first pass and coda, exact K=1 logits, Carry/Bag parameter and compute equality, procedural
  substrate, independent pilot firewall, crossed confirmation, same-checkpoint edge cut, and
  bidirectional swap gates fixed. This successor must determine whether low-rank plasticity caused
  the valid state-formation miss; do not leave the serial-state question closed by the LoRA pilot.
- Cross-program interface probe completed:
  `qwen35_4b_commit_slot_jacobian_value_transport` showed that a fixed latent
  answer slot repairs formatting but its semantic hint remains task/alias
  concentrated (15/48 real versus 11/48 shuffled at 1,024; five mixed tasks
  versus six required). A powered fresh replication must clear task-level
  uncertainty before the interface can license any J-space value work.
- Powered cross-program replication completed:
  `qwen35_4b_commit_slot_semantic_power_replication` fixes the same slot/cap on
  113+113 fresh tasks and requires both task-level ordered-content evidence and
  eight-alias semantic breadth before treating the latent interface as usable.
  Qualification passed with correct support across all 11 targets and choices
  across all 12 aliases; confirmation independently passed with 98/339 ordered
  versus 47/339 shuffled and support across 10/11 targets and all 12 choices.
  The licensed shared J-value measurement was then chance (0.502) and weaker
  than slot margin/non-J state, with midpoint/end phase reversal. Treat the slot
  as a stable output seam, not a stable scalar compiler-state coordinate; causal
  work is not licensed. A post-decision midpoint-only refit reached only 0.538
  versus matched non-J 0.600, so do not replicate this J-axis interpretation.
- Additive J directions do not turn the last native-thought token into a
  hypothesis register. The subsequent explicit semantic anchor writes direct
  aliases but is terminal invalid at an unreachable one-token consequence
  interface, with a fixed composed label map. Retire native J compiler branches;
  a fresh early concrete-text fork must beat late text, duplicate hypotheses,
  and matched-compute sample-more before it can motivate installation.
- Measure the exact behavioral quotient at fresh depth 6 before assuming model-guided pruning is economically
  needed; record wall time, memory, physical transitions, coverage, and selector success.
- If a real search wall appears, test a residualized partial state (feasible parameter domains, materialized
  prefix outputs, per-example target residuals) behind the same within-task AUROC and recall@beam gate.
- Repair the demonstrated visible-only selector gap over exact solver pools (60/60 coverage versus 56/60
  selected) with frozen stability/simplicity/unlabeled-probe rules.
- Replicate the strongest structural compiler results across seeds, lengths, and operator mixes.
- Run a direct-text-program versus typed-bytecode versus latent-slot comparison on one shared task suite.
- Add adversarial paraphrase and compositional splits where direct prompt cues fail.
- Measure whether state-prefix supervision, final-answer supervision, or program-token supervision is the causal lift.
- Build a small diagnostic suite that every compiler-style experiment can run before claiming generalization.

## Required Controls

- Direct answer baseline.
- Same model and data with unstructured output.
- Shuffled or corrupted state supervision when state traces are used.
- Length and family holdouts.

## Stop Conditions

Retire a variant when it improves train or IID accuracy but cannot survive harder length/family/paraphrase splits after two controlled attempts.

Type-only absolute P(viable) at think@256 is retired as a search controller unless a materially richer state
or interface first clears calibration; pooled AUROC alone must not reopen it.
