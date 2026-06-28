# Knowledgebase Protocol

## What Belongs In The Knowledgebase

Put cross-experiment claims in `knowledge/` only when they cite or link to specific experiments. The knowledgebase should make reuse easier without flattening uncertainty.

Good entries:

- a repeated bottleneck observed across experiments,
- a result that changes which direction should be tried next,
- a negative control that prevents a tempting wrong explanation,
- a reusable evaluation pattern.

Avoid entries that are only aspirations or generic advice.

## Evidence Labels

Use these labels in synthesis notes:

- `Confirmed`: directly supported by a result-bearing run.
- `Promising`: supported by a pilot or small run, needs scale or replication.
- `Negative`: tested and failed under the recorded setup.
- `Open`: plausible and not yet adequately tested.

## Updating Generated Files

Do not hand-edit generated files:

- `knowledge/experiment_catalog.md`
- `knowledge/experiment_catalog.csv`
- `knowledge/tag_index.md`
- `knowledge/artifact_index.md`
- `knowledge/source_tracks.md`
- `knowledge/source_tracks.csv`
- `knowledge/experiment_manifest.json`
- `knowledge/readme_coverage.md`

Regenerate them with:

```bash
make catalog
```

