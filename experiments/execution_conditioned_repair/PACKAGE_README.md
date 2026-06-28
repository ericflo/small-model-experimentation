# Execution-Conditioned Repair Experiment Package

This directory is the small, download-friendly package for the completed execution-conditioned repair experiment.

Large artifacts are intentionally stored outside this directory:

`/workspace/large_artifacts/execution_conditioned_repair/`

The original workspace-level `models/` paths are compatibility symlinks to the large-artifact directory, so existing scripts and report-generation commands can still resolve adapter paths such as `models/v2_failure_conditioned_trace_lora`.

## Contents

- `data/`: synthetic datasets, checked datasets, manifests, and SWE-bench direct probe metadata.
- `reports/`: result JSON files, lab notebook, final paper, and legacy report.
- `figures/`: generated figures used in the paper.
- `scripts/`: dataset, training, evaluation, direct SWE-bench probe, and report scripts.
- `src/`: shared repair experiment helpers.
- `configs/`: experiment configuration.
- `requirements.txt`: Python dependencies used by the experiment.

## Main Report

Read `reports/execution_conditioned_repair_paper.md` first.

## Large Artifacts

Adapters and checkpoints were moved to:

`/workspace/large_artifacts/execution_conditioned_repair/models/`

Do not include that directory when downloading only the small package.
