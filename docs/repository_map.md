# Repository Map

## `experiments/`

The primary research library. Every direct child is a standalone experiment. A reader should be able to understand the question, run path, artifacts, and result without needing files from another experiment folder.

Expected files and directories:

- `README.md`: entry point, question, setup, how to run, primary artifacts.
- `metadata.yaml`: generated catalog metadata.
- `experiment_log.md`: chronological notes when available.
- `src/`: local libraries or model/environment code.
- `scripts/`: run, evaluation, and report scripts.
- `configs/`: configs, split manifests, run manifests.
- `data/`: local datasets and records required for the experiment.
- `runs/`: raw run outputs.
- `analysis/`: derived tables, notebooks-as-scripts, summaries, figures.
- `reports/`: final report, paper, html, figures, model cards, adapter notes.

Not every imported experiment has every directory. Missing directories are acceptable when the experiment did not need them.

## `knowledge/`

The compounding memory layer. Generated files make the corpus navigable; human-authored files record interpretation and research strategy.

Key files:

- `synthesis.md`: human cross-track findings.
- `experiment_catalog.md` and `.csv`: generated catalog.
- `tag_index.md`: generated topic index.
- `artifact_index.md`: generated artifact inventory.
- `source_tracks.md` and `.csv`: original raw-track provenance.
- `readme_coverage.md`: imported README coverage and generated stubs.

## `docs/`

Process and operating guidance. Keep this stable and general. Experiment-specific notes belong inside the experiment folder.

## `templates/`

Starting structure for new self-contained experiments.

## `scripts/`

Repository-level maintenance scripts. They must stay dependency-light so future agents can run them immediately.

