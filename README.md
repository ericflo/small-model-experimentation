# Small Model Experimentation

A research operating system for compounding small-model experimentation.

The imported prototype corpus is seed evidence and working examples. It is not the scope of the repository. The repository is designed to support many independent lines of experimentation: new mechanisms, new benchmarks, new selection methods, new posttraining strategies, new tool-use loops, new diagnostics, and new research programs that do not exist yet.

## Start Here

- [research_programs/README.md](research_programs/README.md) for the durable research lines this repo is meant to grow.
- [knowledge/research_program_index.md](knowledge/research_program_index.md) to see which experiments already inform each line.
- [knowledge/program_scorecards.md](knowledge/program_scorecards.md) to choose what to try next and avoid duplicate variants.
- [knowledge/claims/index.md](knowledge/claims/index.md) for structured claims with evidence links.
- [knowledge/synthesis.md](knowledge/synthesis.md) for cross-program claims and current strategic read.
- [knowledge/experiment_catalog.md](knowledge/experiment_catalog.md) for the full experiment inventory.
- [docs/agent_handbook.md](docs/agent_handbook.md) for future agents.
- [docs/experiment_lifecycle.md](docs/experiment_lifecycle.md) before adding a new experiment.
- [docs/research_program_lifecycle.md](docs/research_program_lifecycle.md) before adding a new line of inquiry.
- [docs/artifact_policy.md](docs/artifact_policy.md) before adding large outputs, adapters, or external artifacts.
- [docs/discovery_workflow.md](docs/discovery_workflow.md) to find related programs, claims, and experiments for a rough idea.
- [docs/idea_intake_protocol.md](docs/idea_intake_protocol.md) before turning a rough idea into a run.
- [docs/quality_gates.md](docs/quality_gates.md) for the checks that protect the repository shape.

## Layout

```text
research_programs/<program-id>/
  charter.md             purpose, progress signals, boundaries
  backlog.md             next experiments and controls
  evidence.md            seed evidence and current read

experiments/<experiment-id>/
  README.md              experiment entry point
  metadata.yaml          generated navigation metadata, including programs
  src/ scripts/ configs/ local code and run scaffolding when present
  data/ runs/ analysis/ reports/

knowledge/               generated indexes plus human synthesis and claims
docs/                    operating guidance for people and agents
templates/               starting points for new experiments and programs
scripts/                 repository indexing and validation utilities
```

## Working Rules

Every experiment should either advance an existing research program or justify a new one. Keep experiments self-contained, but connect their results upward into program evidence and shared knowledge.

Create new scaffolds with the repository tools:

```bash
make new-program PROGRAM=multimodal_small_models TITLE="Multimodal Small Models" FOCUS="Test how small models use image, audio, and structured visual evidence."
make new-experiment EXPERIMENT=multimodal_table_probe PROGRAM=multimodal_small_models TITLE="Multimodal Table Probe"
```

Regenerate indexes after changing experiments or programs:

```bash
make check
```

Do not check trained adapter directories into git. Use external artifact manifests for adapters and large model outputs.
