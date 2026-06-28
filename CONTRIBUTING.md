# Contributing

## Add A Research Program

1. Read `research_programs/README.md`.
2. Run `make new-program PROGRAM=<program-id> TITLE="<Title>" FOCUS="<one-sentence focus>"`.
3. Fill in `charter.md`, `backlog.md`, and `evidence.md`.
4. Check the new entry in `research_programs/registry.yaml`.
5. Run `make check`.

A new program should be broad enough to host multiple future experiments and specific enough that progress can be recognized.

## Add An Experiment

1. Pick a research program or create a new one.
2. Pick a stable snake_case experiment id.
3. Run `make new-experiment EXPERIMENT=<id> PROGRAM=<program-id> TITLE="<Title>"`.
4. Fill in the README before running the full experiment.
5. Add a cheap smoke run and record how to run it.
6. Put code, data, runs, analysis, and reports inside the experiment folder.
7. Run `make check`.

## Report A Result

Every result-bearing experiment should answer:

- What program does this advance?
- What question did this test?
- What was the strongest baseline or control?
- What evidence would have falsified the idea?
- What changed after the result?
- What should the next experiment do differently?

## Artifact Policy

Keep small reproducible artifacts in git. Do not check trained adapter directories into git. Put externally stored artifacts in a manifest that names the path, checksum when available, and how to regenerate or retrieve them.
