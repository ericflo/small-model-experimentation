# Qwen3.5-4B Commit-Slot Semantic Power Replication

This study tests whether the parent experiment's fixed-1,024 ordered-thought
advantage over shuffled thought is a task-general semantic effect rather than a
five-task, alias-concentrated near miss.

## Research Program

- Primary: `interpretability_and_diagnostics`.
- Secondary: `test_time_reasoning_budget` and
  `structured_execution_and_compilers`.
- Direct parent: `qwen35_4b_commit_slot_jacobian_value_transport`, terminal
  `COMMIT_SLOT_SEAM_FAIL`.
- J mechanism anchor: `qwen35_4b_jacobian_transport_control_replication`.

## Question

At one fixed 1,024-token thought budget, does ordered native thought reliably
improve the next semantic alias choice over both an immediate slot and an exact-
length permutation of the same thought tokens, across enough fresh task and
alias units to support a later J-space value experiment?

## Parent evidence and hypothesis

The parent repaired answer mode: an alias was the unmasked top token on 41/48
long traces. Ordered thought scored 15/48 versus the equivalent 12/48 no-thought
and 11/48 shuffled. It passed both pooled gap gates but had five mixed tasks
versus six required; task-bootstrap intervals crossed zero and effects were
alias concentrated. Post-hoc bias subtraction did not improve the slot.

The narrow hypothesis is that the +8.33pp ordered-over-shuffled task effect is
real but underpowered. This replication fixes cap 1,024, expands each seam stage
to 113 tasks (339 traces), balances all 11 target operations, and requires both
task-level uncertainty and semantic-support gates. It does not change syntax,
aliases, decoding, model, task family, or the three-trace policy.

## Setup

- Only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Exact replicated 24-coordinate lens, SHA-256
  `e373b6e93956fdfc5cb446e9bee8249655707c8258a7868f0653d11f1ffd0213`,
  frozen layers 4--8. It remains unused unless the seam replicates.
- Transformers bf16 SDPA, unpadded batch one; cached native generation and exact
  cache-free full-prefill slot/control logits.
- 322 new exact-depth-two procedural tasks: 113 semantic qualification, 113
  untouched semantic confirmation, 48 value fit, and 48 causal confirmation.
- All visible sets have one identifiable first-operation type and no depth-one
  fit; fingerprints are unique and disjoint from five direct parents.
- Fixed cap 1,024; three traces/task; temperature 0.6, top-p 0.95, top-k 20.
- Policy: append exactly `</think>\n\nFirst:` and take argmax over the 12 public
  one-token aliases.
- Controls: immediate no-thought slot, deterministic exact-token-multiset
  shuffle, unmasked full-vocabulary logits, and same-prefix close-only free-form
  output.

The slot is a constrained deployment interface. It supplies syntax and a closed
vocabulary, never answer identity.

## Power and frozen gates

The parent task-level ordered-minus-shuffled mean was 0.08333 with SD 0.35486.
A one-sided alpha-0.05 normal planning approximation requires 113 task units for
80% power; both seam stages use exactly 113. The actual decision uses a
nonparametric task bootstrap, not the approximation.

Each stage independently requires:

- real slot accuracy in 20%--70%;
- at least 28 tasks with both correct and incorrect real traces;
- at least +3pp over no-thought and +5pp over shuffled thought;
- one-sided 95% task-bootstrap lower bound above zero for real minus shuffled;
- correct successes spanning at least eight target aliases;
- at least eight distinct chosen aliases;
- unmasked top-is-alias rate at least 75% and mean alias mass at least 50%; and
- 100% finite real rows, with every evaluated control finite.

Selection tests only fixed cap 1,024. If it passes, one untouched 113-task
confirmation must satisfy the identical gates. Splits may not be pooled to
rescue a miss. Only `POWERED_COMMIT_SLOT_SEAM_REPLICATED` may reopen value-code
implementation; all J/value/control/causal commands currently fail closed.

## Run

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  experiments/qwen35_4b_commit_slot_semantic_power_replication/tests -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/qwen35_4b_commit_slot_semantic_power_replication/scripts/run.py \
  --stage smoke
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/qwen35_4b_commit_slot_semantic_power_replication/scripts/power_audit.py
```

After anchoring the design boundary:

```bash
.venv/bin/python experiments/qwen35_4b_commit_slot_semantic_power_replication/scripts/run.py --stage model-smoke
.venv/bin/python experiments/qwen35_4b_commit_slot_semantic_power_replication/scripts/run.py --stage seam-selection
.venv/bin/python experiments/qwen35_4b_commit_slot_semantic_power_replication/scripts/run.py --stage seam-confirmation
```

## Status

Design and adversarial review are complete before any model call. CPU smoke
passes 322/322 unique fresh exact-depth tasks, zero overlap with five parents,
balanced target support, exact lens hash, and reachable gates. The power receipt
records 113 tasks required and planned per seam stage, approximate power 0.8027,
and the exact parent diagnostic hash. Model and scientific outcomes are unopened.

## Scope

A seam pass would replicate constrained semantic elicitation, not J value and
not installed capability. Gold labels score gates. A later J stage would remain
oracle until a label-free controller beats frozen inference and matched-compute
sampling on untouched procedural tasks.

## Knowledgebase Update

- Update all three program ledgers and shared synthesis at terminal gates.
- Reserve no claim ID while the claim re-grade remains open.

## Artifacts

- `assets/context_lens.pt`: byte-identical mechanism anchor.
- `data/procedural/`: four frozen fresh splits and manifest.
- `runs/smoke/`: CPU, reachability, and power receipts.
- `reports/preregistration.md` and `reports/design_review.md`: immutable rules.
- `scripts/run.py`: fixed-cap seam harness; later stages fail closed.
