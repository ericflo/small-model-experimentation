# Backlog

## Next Experiments

- The LoRA architecture counterfactual completed with valid `PILOT_MECHANISM_MISS`; the held-fixed,
  zero-initialized 892M-parameter full-rank successor also completed, with raw joint-state accuracy
  0.00277 versus the 0.40 gate, Carry minus Bag -0.0156, negative unseen-K scaling, and noncausal
  swaps. Its analyzer emitted `PILOT_STATE_FORMATION_MISS`, but post-result preregistration audit
  assigns `PILOT_PROMOTION_BLOCKED`: the run simultaneously failed the non-capacity requirements of
  positive Carry-minus-Bag and positive query-kind effects. It therefore does not isolate or close
  the rank/capacity question. Do not run either existing experiment's confirmation, edge-cut, or
  sample-more stages, and do not open the interface successor.
- Run the mandatory fresh RNG-matched three-seed state-formation adjudication, pairing rank-32 LoRA
  and full-rank deltas under matched generated rows, initialization/training randomness, inference
  randomness, and identical state readouts. Use it to decide whether adapter capacity prevented the
  deeper representation from forming. Defer any conclusion that future serial-latent recurrence
  work must use a materially new representation or supervision design until that adjudication.
  Retain an equal-compute noncarrying control, the exact untouched K=1 path, and an early registered
  state-decoding positive control before expensive extrapolation or causal work.
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
  hypothesis register. The late explicit anchor was invalid, and early
  concrete text redirected direct execution across 24 operations but failed
  its interface. The first materialized-residual successor then ended without
  a durable, authenticated model result: one preflight abort and one terminal
  52-row write-order/EOS-receipt incident. Retire opaque-name timing variants.
  Resume materialized residuals only in a new experiment with fresh tasks/
  record IDs/seeds,
  write-before-authentication quarantine, the exact model/tokenizer EOS pair,
  and the already frozen candidate-blind, shuffled, exhaustive, and sampled/
  logical-token-matched controls. That identity is now reserved as
  `qwen35_4b_materialized_residual_sibling_search_fresh_replication`; fresh
  construction and adversarial review remain the next gates.
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
