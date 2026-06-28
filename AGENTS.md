# Agent Operating Guide

This repository is a compounding research system. The current experiments are examples and seed evidence; your job is to extend the frontier while preserving what has already been learned.

## First Pass

1. Read [research_programs/README.md](research_programs/README.md).
2. Read [knowledge/research_program_index.md](knowledge/research_program_index.md).
3. Read [knowledge/program_scorecards.md](knowledge/program_scorecards.md).
4. Read [knowledge/claims/index.md](knowledge/claims/index.md).
5. Read [knowledge/synthesis.md](knowledge/synthesis.md).
6. Use [knowledge/experiment_catalog.md](knowledge/experiment_catalog.md) and [knowledge/tag_index.md](knowledge/tag_index.md) to find close prior work.
7. Before adding work, read [docs/idea_intake_protocol.md](docs/idea_intake_protocol.md), [docs/experiment_lifecycle.md](docs/experiment_lifecycle.md), [docs/research_program_lifecycle.md](docs/research_program_lifecycle.md), and [docs/knowledgebase_protocol.md](docs/knowledgebase_protocol.md).
8. Use [docs/quality_gates.md](docs/quality_gates.md) to understand what `make check` enforces.

## Non-Negotiables

- Treat the imported tracks as prototypes, not as the repo boundary.
- Keep experiments self-contained under `experiments/<id>/`.
- Attach every new experiment to a research program, or create a new program.
- Use an idea intake note for material new directions and name the closest near-duplicate before running.
- Preserve negative results and failed controls; they are part of the map.
- Update program evidence and shared synthesis when a result changes strategy.
- Run `make check` before committing.

## When Starting A New Experiment

Pick the program first. If no program fits, create one with:

```bash
make new-program PROGRAM=<program_id> TITLE="<Title>" FOCUS="<one-sentence focus>"
```

Then create the experiment with:

```bash
make new-experiment EXPERIMENT=<experiment_id> PROGRAM=<program_id> TITLE="<Title>"
```

Fill in the README, make the smallest runnable smoke path real, and only then run expensive work.

## When Starting A New Program

Create `research_programs/<program-id>/` with a charter, backlog, and evidence ledger. Add it to `research_programs/registry.yaml`; `make new-program` does the mechanical pieces. The charter must explain why the program is not just a variant of an existing line.

## When Editing Imported Work

Prefer additive notes in `README.md`, `experiment_log.md`, `analysis/`, or `reports/`. Do not rewrite historical outputs unless you are regenerating them from code and can explain the change.
