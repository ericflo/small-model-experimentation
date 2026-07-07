# Does the structure-proposal wall exist in language? Yes — induction persists as a wall (unlike simulation)

## Motivation
C37 showed the model SIMULATES multi-step reasoning in language near-perfectly (no depth-3 wall). But that tests
C13-style simulation, not the C32/C36 structure-**proposal** wall. This asks the complementary question: can the
model **induce** a hidden compositional rule from examples in language, or does the proposal wall persist even in
its native domain?

## Method
Relational-composition **induction**: R=4 made-up relations (each a random bijection over ~16 made-up entities);
a hidden depth-D rule = a fixed sequence of D relations. Give the full relation KB + k examples (start → answer
applying the hidden rule) + a query start **not** among the example starts → the model must infer which relations
compose and apply. **Min-depth-verified** (reject shorter-equivalent rules), **uniqueness-pinned** (examples
uniquely determine the rule), contamination-free. **Critical control (review-mandated): an application-only arm**
(rule GIVEN explicitly) — this multi-relation substrate is harder to *execute* than C37's chains, so induction
failure is meaningful only where application is easy. Renderings linguistic-symbolic (primary); no-think + think
(budget 4096, truncation-checked).

## Result
| condition | d1 | d2 | d3 | d4 |
|---|---|---|---|---|
| **application** (execute given rule), no-think | **0.86** | 0.28 | 0.46 | 0.12 |
| application, think | 0.75 | — | — | — |
| **INDUCTION** (infer rule), no-think | **0.00** | 0.04 | 0.08 | 0.02 |
| INDUCTION, think (budget 4096, no truncation) | **0.50** | — | — | — |

(guess baseline ≈ 0.06; C37 linguistic *simulation* = 0.99 no-think at depth-3.)

- **Clean forward-pass dissociation at depth-1** (where application is easy, so induction is isolable): the model
  **executes** a given relational rule (0.86) but **cannot infer** one from examples (0.00 = chance) in a single
  forward pass. Induction is at chance no-think at **all** depths.
- **Thinking only partially rescues induction** (0.00 → 0.50 at depth-1, budget 4096, verified no truncation — the
  reasoning is correct but error-prone), still far below application (0.75 with think) and far below C37's
  linguistic simulation (0.99).
- So the model is an **executor, not an inducer**, in language as in formal domains — corroborating C32/C36
  (value-computer, not structure-proposer) as a **cross-modality** property.

## Implication (C37 + C38 together)
The compositional wall has **two components that dissociate by modality**:
- **Simulation / execution** is modality-**dependent** — the formal wall (depth-3, C13–C36) vanishes in language
  (C37).
- **Proposal / induction** is modality-**general** — hard in both formal (C32/C36) and language (this) — the
  deeper, more fundamental limit.

The model reasons multi-step in language, but it does **not induce rules**. The structure-proposal wall is the one
part of the whole arc that holds even in the model's native domain.

## Honest scope & caveats
- This multi-relation substrate's **application itself degrades at depth 2+** (0.28/0.46/0.12 — the small model
  struggles to chain 2–3 full-bijection lookups no-think), so induction is cleanly isolable only at **depth-1**;
  but induction is already at chance there. Deeper induction was not cleanly measured (application confound + very
  slow think runs).
- Think depths 2–4 not completed (budget-4096 think is ~35 min/condition); d1 think (0.50) is the clean think
  point. Single seed; n=24–50.
- Formal-dict rendering (code-mode confound, C37) not used as primary; linguistic-symbolic is primary.

## Artifact Manifest
See `reports/artifact_manifest.yaml`.
