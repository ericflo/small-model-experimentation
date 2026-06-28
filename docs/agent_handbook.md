# Agent Handbook

## Mission

This repository exists to make small-model research compound. Your job is not only to run another experiment; it is to make the next experiment smarter because this one exists.

## Before You Change Anything

1. Identify the relevant prior experiments through `knowledge/experiment_catalog.md` and `knowledge/tag_index.md`.
2. Read at least the local README and primary report for each close prior.
3. Write down the new uncertainty your work targets.
4. Decide what result would falsify the idea.

## How To Add Value

- Prefer experiments that distinguish between plausible explanations.
- Keep controls close to the main result.
- Record negative results with the same care as positive ones.
- Make result tables easy to compare across methods.
- Link follow-up ideas to specific evidence, not just intuition.

## Edit Discipline

Do not rewrite old experiment outputs to make a cleaner story. Add an interpretation note, a corrected report, or a new analysis artifact that says what changed. Historical records are useful precisely because they show the path.

## Completion Checklist

- Experiment folder is self-contained.
- README states the question, setup, run command, result, and artifacts.
- Reports separate deployable evidence from oracle/hidden evaluation.
- Controls and ablations are named clearly.
- `make catalog` and `make validate` pass.
- `knowledge/synthesis.md` or another knowledge page is updated when the result changes future priorities.

