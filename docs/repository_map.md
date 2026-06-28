# Repository Map

## `research_programs/`

The forward-looking program layer. Each direct child is a durable line of inquiry with a charter, backlog, and evidence ledger. Programs are how the repository grows beyond the imported seed tracks.

Expected files:

- `charter.md`: purpose, progress signals, boundaries.
- `backlog.md`: concrete next experiments and controls.
- `evidence.md`: seed experiments, claims, and current read.

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

- `synthesis.md`: human cross-program findings.
- `research_program_index.md` and `.csv`: generated program-to-experiment index.
- `program_scorecards.md`: hand-authored next-experiment and anti-duplication scorecards.
- `decision_records/`: strategic decisions that affect more than one experiment.
- `experiment_catalog.md` and `.csv`: generated catalog.
- `experiment_readiness.md` and `.csv`: generated per-experiment curation matrix for README, report, log, run-surface, smoke-command, artifact-manifest, and program-assignment readiness.
- `future_experiment_queue.json`: structured future-work source covering concrete probes, infrastructure tasks, and candidate program lines.
- `future_experiment_queue.md` and `.csv`: generated views of the future queue.
- `tag_index.md`: generated topic index.
- `artifact_index.md`: generated artifact inventory.
- `artifact_manifest_index.md` and `.csv`: generated index of dataset, checkpoint, large-artifact, and reproducibility manifests.
- `source_tracks.md` and `.csv`: original raw-track provenance.
- `readme_coverage.md`: imported README coverage and generated stubs.

## `docs/`

Process and operating guidance. Keep this stable and general. Experiment-specific notes belong inside the experiment folder.

## `templates/`

Starting structure for new self-contained experiments, research programs, idea intake notes, and decision records. Prefer the scaffold scripts over manual copying for experiments and programs so registry and metadata stay aligned.

## `scripts/`

Repository-level maintenance scripts. They must stay dependency-light so future agents can run them immediately.

- `build_knowledgebase.py`: regenerate generated catalogs and indexes.
- `validate_repository.py`: check repository invariants.
- `check_markdown_links.py`: check local links in navigation and knowledge surfaces.
- `check_python_syntax.py`: compile maintenance scripts without cache artifacts.
- `check_repository_text.py`: scan for stale framing and temporary scaffold residue.
- `find_related.py`: route rough ideas to related programs, claims, and experiments.
- `scaffold_research_program.py`: create a new program directory and registry entry.
- `scaffold_experiment.py`: create a new experiment attached to one or more programs.
