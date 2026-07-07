# Agent Handbook

## ⚠️ Current priority: claim-ledger re-grade

An adversarial review (2026-07-06) verified **102 findings** across the 41 claims:
the data is sound but the **statuses, headline numbers, and superlatives are
inflated**, and several corrections never propagated. Before adding new claims,
work the checklist in [`knowledge/claims/claim_review_todo.md`](../knowledge/claims/claim_review_todo.md)
(full review: [`knowledge/claims/claim_review.md`](../knowledge/claims/claim_review.md)).
These are corrections — downgrades, scope-narrowing, number fixes — not
retractions; the underlying data mostly supports the softened versions. Do the
HIGH items first (over-issued `Confirmed` on C1/C6/C7/C8/C36; "p<0.01" on
1–5-task differences; think/oracle numbers standing in for a ~0 deployable
metric; stale cross-references and a self-contradicting synthesis).

## Mission

This repository exists to make small-model research compound across many research programs. Your job is not only to run another experiment; it is to make the next experiment and the next research line smarter because this one exists.

## Before You Change Anything

1. Identify the relevant program through `research_programs/README.md` and `knowledge/research_program_index.md`.
2. Identify close prior experiments through `knowledge/experiment_catalog.md` and `knowledge/tag_index.md`.
3. Read the local README and primary report for each close prior.
4. Write down the new uncertainty your work targets.
5. Decide what result would falsify the idea.

## How To Add Value

- Prefer experiments that distinguish between plausible explanations.
- Prefer ideas that advance or create durable research programs.
- Use `make new-program` and `make new-experiment` for new scaffolds so metadata, registry entries, and smoke paths start aligned.
- Keep controls close to the main result.
- Record negative results with the same care as positive ones.
- Make result tables easy to compare across methods.
- Link follow-up ideas to specific evidence, not just intuition.

## Edit Discipline

Do not rewrite old experiment outputs to make a cleaner story. Add an interpretation note, a corrected report, or a new analysis artifact that says what changed. Historical records are useful precisely because they show the path.

## Completion Checklist

- Experiment folder is self-contained.
- Owning research program is named.
- README states the question, setup, run command, result, and artifacts.
- Reports separate deployable evidence from oracle/hidden evaluation.
- Controls and ablations are named clearly.
- `make check` passes.
- Owning program evidence/backlog and `knowledge/synthesis.md` or a claim page are updated when the result changes future priorities.
