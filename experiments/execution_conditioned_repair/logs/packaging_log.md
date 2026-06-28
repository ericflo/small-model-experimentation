# Packaging Log

## 2026-06-20

The completed execution-conditioned repair experiment was organized into the common experiment layout.

Actions:

- Copied small artifacts from workspace-level `data/`, `reports/`, `figures/`, `scripts/`, `src/`, `configs/`, `requirements.txt`, and `README.md` into `/workspace/experiments/execution_conditioned_repair/`.
- Moved model adapters and checkpoints from workspace-level `models/` into `/workspace/large_artifacts/execution_conditioned_repair/models/`.
- Replaced each workspace-level model directory with a symlink to the corresponding large-artifact path so existing scripts keep working.

Size after packaging:

- Small package: about 13 MB.
- Large artifacts: about 13 GB.

Download guidance:

- Download `/workspace/experiments/execution_conditioned_repair/` for paper, data, scripts, logs, and compact result artifacts.
- Download `/workspace/large_artifacts/execution_conditioned_repair/` only if adapter weights and checkpoints are needed.
