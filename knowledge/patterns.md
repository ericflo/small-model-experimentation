# Patterns

## Program Before Experiment

Start by naming the research program and the uncertainty it advances. If no program fits, create a new one before running the experiment.

Use when:

- the idea could become a durable line,
- the result should change shared strategy,
- several follow-up experiments are plausible.

## Candidate Pool Then Selector

Generate multiple candidate answers, programs, traces, or retrieved adaptations. Measure oracle coverage separately from deployable selection. This pattern is central when direct sampling leaves a residual set but hidden-correct candidates exist somewhere in the pool.

Use when:

- public evidence is sparse,
- candidate generation looks promising,
- the failure mode is false visible-pass commits.

## Executable Intermediate

Ask the model to produce or condition on something executable: code, a transform function, typed bytecode, latent slots, operator choices, or state traces. Then evaluate the intermediate directly.

Use when:

- direct final answers are brittle,
- a visible example can execute the candidate,
- a typed or structured representation can be supervised.

## Oracle Ceiling With Deployable Gap

Report the hidden-oracle ceiling as a diagnostic, but make the deployable selector the headline if the question is practical deployment.

Use when:

- a hidden-correct candidate exists,
- public evidence cannot safely pick it,
- the next experiment should target evidence rather than generation.

## Control-Clean Gain

A gain matters more when it beats random, shuffled, corrupted, frozen, or order-only controls. A result that does not beat controls should become a negative finding, not a buried run.

## Self-Contained Experiment

Every experiment owns its code, data, reports, and local analysis. Shared knowledge is extracted upward into `knowledge/`; shared machinery is promoted only after repeated independent need.

## Claim Ledger Update

When a result changes what future work should believe, update a claim file in `knowledge/claims/` or add one. Claims should link to evidence and state whether they are confirmed, promising, negative, open, or retired.
