# Backlog

## Next Experiments

- Cross-program interface probe in progress:
  `qwen35_4b_commit_slot_jacobian_value_transport` asks whether a fixed latent
  answer slot exposes semantic choice after native thought. It must beat the
  same slot with no thought and shuffled thought before the interface can
  license any J-space value work; constrained choice is not free-form execution.
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
