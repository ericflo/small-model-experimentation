# Backlog

## Next Experiments

- Architecture-counterfactual setup ready: `qwen35_4b_state_carry_vs_state_bag`
  repeats two full-width Qwen3.5 hybrid motifs and compares one inherited latent
  state against an equal-parameter/equal-decoder-layer-token bag of reset shallow states. It
  has a fresh query-after-state substrate, an independent pilot seed/firewall,
  K=1 parity, crossed task×seed inference, unseen-K/depth extrapolation,
  same-checkpoint edge cuts, bidirectional geometry-matched state swaps, a joint
  holdout gate, and interface-qualified matched-layer-token-budget explicit-CoT sampling.
  No model result exists yet; the next action is fresh CPU regeneration/tests,
  then the live 48 GiB Ada model smoke followed by the seed-7401 paired pilot—not
  an expensive full sweep. Mixed semantic echo was removed and requires a fresh
  successor if continuous state proves readable but unused.
- Conditional capacity successor: if a valid rank-32 LoRA outcome fails to establish
  deep state formation, create and execute a fresh experiment that
  replaces extra-call LoRA with zero-initialized full-rank deltas on layers 12–19.
  Keep the base first pass/coda frozen and K=1 exact. This is mandatory to resolve
  whether low rank, rather than serial state, caused the negative. Mechanics/data
  failures and infeasible gates are not capacity evidence; a sample-more-only loss
  or strongly readable-but-unused state means LoRA already formed the representation,
  with the latter routed to the controlled interface successor.
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
  Use this interface for task-held-out value measurement, while treating strong
  target-identity heterogeneity as an explicit nuisance rather than a compiler
  capability result.
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
