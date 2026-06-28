# Evidence-Conditioned Selection

## Purpose

Turn candidate pools into deployable decisions. This program studies how to choose, abstain, rerank, verify, or gather more evidence when hidden-correct candidates exist but public evidence is weak.

## Why This Is A Program

Many imported experiments found that generating useful candidates is easier than selecting them safely. That pattern should become a whole research line, not a footnote inside generation experiments.

## Progress Signals

- False visible-pass commits decrease at fixed or improved recall.
- Selectors separate oracle-only coverage from deployable evidence.
- Abstention improves precision without hiding failure rates.
- Selection policies transfer across tasks, families, and candidate generators.

## Boundaries

Candidate generators can come from any program. This line owns the decision layer and the evidence needed to make that decision without hidden labels.
