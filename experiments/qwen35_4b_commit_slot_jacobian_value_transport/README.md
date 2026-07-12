# Qwen3.5-4B Commit-Slot Jacobian Value Transport

This study tests whether a syntax-only forced answer slot reveals a usable
alias decision and, if so, whether a scalar Jacobian certainty coordinate can
improve that decision.

## Research Program

- Primary: `interpretability_and_diagnostics`.
- Secondary: `test_time_reasoning_budget` and
  `structured_execution_and_compilers` (the fixed latent answer interface).
- Direct parent: `qwen35_4b_forced_commit_jacobian_value_transport`.
- Earlier seam parents: `qwen35_4b_native_thought_jacobian_value_transport` and
  `qwen35_4b_native_thought_seam_budget_ladder`.
- Mechanism anchor: `qwen35_4b_jacobian_transport_control_replication`.

## Question

After a fixed thought budget, can the controller append only
`</think>\n\nFirst:` and read a correct next choice from the 12 public alias
tokens often enough to support value measurement? If yes, is correct-alias
probability held-out-by-task decodable and causally writable as one task-general
J-space coordinate rather than answer identity or ordinary slot confidence?

## Hypothesis

Close-only forced answers failed because the model restarted analysis instead
of entering the requested output grammar. Supplying the grammar but not the
alias should isolate the semantic choice: the constrained next-token argmax may
recover information already present in the thought prefix. Real thought must
beat an immediate no-thought slot and retain correct/incorrect variation. A
learned scalar value coordinate must then beat direct correct-alias activity,
ordinary slot margin, shuffled labels, exact random, direct-output-gradient,
and non-J controls.

## Setup

- Only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Exact replicated 24-coordinate lens, SHA-256
  `e373b6e93956fdfc5cb446e9bee8249655707c8258a7868f0653d11f1ffd0213`,
  frozen layers 4--8.
- Transformers bf16 SDPA, unpadded batch one; cached native trace generation and
  exact full-prefill slot logits.
- 96 new exact-depth-two procedural tasks: 16 slot selection, 16 untouched slot
  confirmation, 32 value fit, 32 causal confirmation.
- Every visible set has one identifiable first-operation type and no depth-one
  solution; all fingerprints are unique and disjoint from four parents.
- Three trace samples/task, temperature 0.6, top-p 0.95, top-k 20.
- Paired caps 256/512/1024. At each cap, append close plus fixed slot text and
  take argmax over the 12 public one-token aliases.
- Controls: immediate no-thought slot, an exact-length deterministic permutation
  of the same thought-token multiset, unconstrained full-vocabulary logits, and
  close-only free-form output from the exact same real trace prefix.

The controller supplies syntax and a closed public answer vocabulary, never the
correct alias. This is a constrained deployment interface, not natural output
and not directly comparable to free-form accuracy.

## Frozen Gates

Selection requires constrained slot accuracy in 20%--80%, at least six tasks
with both correct and incorrect traces, 100% finite rows, at least +5pp over the
task-level no-thought slot, and +3pp over exact-length shuffled thought. Freeze
the smallest passing cap. Untouched confirmation repeats the range/count/finite
gates and requires +3pp over no-thought and +2pp over shuffled thought.
Close-only free-form and unmasked alias mass are diagnostics, not rescue arms.

Only `COMMIT_SLOT_SEAM_REPLICATED` opens value fitting. Deterministic
correct-alias probability at 0.5 and 1.0 of the cap is the oracle value label.
J value at the causally earlier final-thought position must reach
held-out-by-task pairwise AUC 0.65, beat correct-alias coordinate activity by
0.03 and ordinary slot margin by 0.02, and leave a within-task shuffled null near
chance. Only that pass may open exact per-prefix bf16 controls and causal tests.

## Run

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/qwen35_4b_commit_slot_jacobian_value_transport/scripts/run.py \
  --stage smoke
```

After the design boundary is anchored:

```bash
.venv/bin/python experiments/qwen35_4b_commit_slot_jacobian_value_transport/scripts/run.py --stage model-smoke
.venv/bin/python experiments/qwen35_4b_commit_slot_jacobian_value_transport/scripts/run.py --stage seam-selection
.venv/bin/python experiments/qwen35_4b_commit_slot_jacobian_value_transport/scripts/run.py --stage seam-confirmation
```

Value/control/causal stages fail closed until their audited implementations are
committed after a replicated slot seam.

## Status

Design and adversarial review complete before any model call. CPU smoke passes:
96 unique fresh exact-depth tasks, zero overlap with four parents, exact lens
hash, reachable gates, and a corrected seed block after the first CPU attempt
caught an exact prior-split generator-seed collision. Scientific outcomes are
unopened.

## Scope

The slot and alias mask are explicit deployment scaffolds. They may elicit a
choice, but do not install a general free-form capability. Correct-label value
and donor selection are oracle. Capability requires a later non-oracle method
to beat frozen, close-only/free-form, no-thought slot, and matched sampling on
untouched contamination-free tasks.

## Knowledgebase Update

- Update both program ledgers and synthesis at each terminal gate.
- No claim ID while the repository claim re-grade remains open.

## Artifacts

- `assets/context_lens.pt`: byte-identical replicated lens.
- `data/procedural/`: four frozen splits and manifest.
- `runs/smoke/`: CPU/gate receipts.
- `reports/preregistration.md` and `reports/design_review.md`: immutable rules;
  both prose and the semantic config payload are design-commit hash-anchored.
