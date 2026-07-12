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

Terminal seam decision: `POWERED_COMMIT_SLOT_SEAM_REPLICATED`. Qualification
and its equally powered untouched confirmation independently passed every
frozen gate. All 678 paths remained open to the fixed 1,024 cap and all
row/control contracts passed.

| frozen metric | qualification | confirmation | gate |
| --- | ---: | ---: | ---: |
| real slot accuracy | 92/339 (27.14%) | 98/339 (28.91%) | 20%--70% |
| no-thought accuracy | 11/113 (9.73%) | 8/113 (7.08%) | real minus >=3pp |
| shuffled-thought accuracy | 46/339 (13.57%) | 47/339 (13.86%) | real minus >=5pp |
| one-sided task lower, real minus shuffled | +8.85pp | +9.44pp | >0 |
| mixed tasks | 32/113 | 31/113 | >=28 |
| correct / chosen alias support | 11 / 12 | 10 / 12 | >=8 / >=8 |
| unmasked top-is-alias / alias mass | 88.20% / 66.79% | 87.61% / 66.35% | >=75% / >=50% |

The parent hint therefore independently generalizes twice across 226 fresh task
units. Ordered thought contributes answer-relevant information beyond an
identical shuffled token multiset and the syntax-only no-thought slot. A
deterministic post-decision audit gives separate two-sided task-bootstrap
intervals of [7.96pp, 19.17pp] and [8.26pp, 21.83pp]; pooling is descriptive
only. Correct-answer mention strata do not explain the effect.

The remaining identity nuisance is load-bearing: confirmation had no successful
`horse` target rows, while `tiger` and `river` were favored by shuffle. The seam
is replicated, but any J/value model must be task-held-out and prove incremental
value beyond correct-alias activity, ordinary slot margin, and alias identity.
J/value/control/causal implementation remains unavailable until a new
outcome-blind adversarial audit and code boundary are committed.

## Scope

This pass replicates constrained semantic elicitation, not J value and not
installed capability. Gold labels score gates. A later J stage remains a
measurement/oracle stage until a label-free controller beats frozen inference
and matched-compute sampling on untouched procedural tasks.

## Knowledgebase Update

- Update all three program ledgers and shared synthesis at terminal gates.
- Reserve no claim ID while the claim re-grade remains open.

## Artifacts

- `assets/context_lens.pt`: byte-identical mechanism anchor.
- `data/procedural/`: four frozen fresh splits and manifest.
- `runs/smoke/`: CPU, reachability, and power receipts.
- `reports/preregistration.md` and `reports/design_review.md`: immutable rules.
- `reports/pre_selection_implementation_audit.md`: outcome-blind code audit.
- `reports/post_confirmation_adversarial_audit.md`: post-decision scope and
  nuisance audit.
- `scripts/run.py`: fixed-cap seam harness; later stages fail closed.
- `runs/seam_selection*.json*` and `runs/seam_confirmation*.json*`: complete,
  hash-locked passing stages.
- `analysis/analyze_replication.py` and `analysis/replication_audit.json`:
  deterministic stagewise and descriptive cross-stage audit.
