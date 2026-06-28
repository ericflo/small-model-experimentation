# Agent Handbook

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
- `make catalog` and `make validate` pass.
- Owning program evidence/backlog and `knowledge/synthesis.md` or a claim page are updated when the result changes future priorities.
