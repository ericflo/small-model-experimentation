# Small Model Experimentation

A unified research repository for two intensive tracks of small-model experimentation. The raw track dumps have been normalized into one self-contained experiment library, with shared indexes and practices that make future work build on what is already known.

## Start Here

- [knowledge/synthesis.md](knowledge/synthesis.md) for the cross-track findings and research direction.
- [knowledge/experiment_catalog.md](knowledge/experiment_catalog.md) for a searchable table of every experiment.
- [knowledge/tag_index.md](knowledge/tag_index.md) to browse by topic.
- [docs/agent_handbook.md](docs/agent_handbook.md) for future agents.
- [docs/experiment_lifecycle.md](docs/experiment_lifecycle.md) before adding a new experiment.

## Layout

```text
experiments/<experiment-id>/
  README.md              experiment entry point
  metadata.yaml          generated navigation metadata
  src/                   local code when present
  scripts/               runnable experiment/report scripts when present
  configs/               configs and manifests when present
  data/                  local datasets or generated records when present
  runs/                  run outputs when present
  analysis/              summaries, derived tables, figures
  reports/               final reports, papers, html, figures

knowledge/               generated indexes plus human synthesis
docs/                    operating guidance for people and agents
templates/experiment/    starting point for new experiments
scripts/                 repository indexing and validation utilities
```

## Working Rules

Each experiment should remain self-contained. Shared knowledge belongs in `knowledge/`; shared process belongs in `docs/`; reusable starting structure belongs in `templates/`. Do not collapse experiments into a single shared source tree unless several completed experiments prove the abstraction is stable.

Regenerate indexes after changing experiments:

```bash
make catalog
make validate
```

Large model artifacts should use Git LFS. The repository already tracks common checkpoint extensions such as `*.safetensors`, `*.pt`, `*.ckpt`, and `*.bin`.

