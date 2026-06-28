# Backlog

## Next Experiments

- Compare prompt-memory, constraint-memory, test-memory, and candidate-memory using the same retrieved items.
- Build a small memory schema for verified skills, failure cases, invariants, and retrieval diagnostics.
- Measure same-family retrieval against random, corrupted, and shuffled controls.
- Use memory to generate counterexamples rather than direct answers.
- Study when retrieved analogies hurt and how to detect that before committing.

## Required Controls

- Random retrieval.
- Corrupted retrieval.
- Shuffled query retrieval.
- Direct no-memory baseline.

## Stop Conditions

Do not promote a memory mechanism that fails to beat direct inference and controls on strict task metrics.
