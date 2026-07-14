# Qwen3.5-4B Tokenizer-EOS Answer Commit Factorial Experiment Log

## 2026-07-14

- Created only after the predecessor published terminal
  `NO_VALID_RESIDUAL_ANSWER_SEAM` and passed adversarial interpretation review.
- Registered the predecessor as the closest near-duplicate and changed only
  the answer-stage stopping hypothesis on fresh future identities.
- Added a model-free first-stop/strict-precommit smoke with HF-EOS, early,
  interior/repeated, missing, and extra-precommit controls. No model was loaded
  or called; live execution remains held pending adversarial design review.
- Independent adversarial review returned `HOLD_DESIGN`: boundary pairing was
  not fail-closed, thinking was not shared across all four continuation cells,
  grammar/config arity disagreed, and conditional mechanics was adaptable.
- Prospectively froze all 192 causal pairs, one persisted thought per task,
  arity-parametric token grammar, new namespaces/seeds, full 24-task mechanics
  geometry, resource matching, inference, terminal outcomes, and lock order.
- Extended the zero-call smoke to accept tokenizer-first, HF-first, and shared-
  cap paired traces while rejecting divergence and false length claims. Live
  calls remain held pending independent rereview and implementation review.

## Scaffold

Created as a new experiment scaffold.
