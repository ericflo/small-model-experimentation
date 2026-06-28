# Counterexample Rule Repair Experiment Package

This directory is the small, download-friendly package for the counterexample-to-rule repair experiment.

Large artifacts are intentionally stored outside this directory:

`/workspace/large_artifacts/counterexample_rule_repair/`

## Contents

- `configs/`: experiment configuration.
- `data/`: train, validation, all-record JSONL files, and the dataset manifest.
- `figures/`: generated figures used in the paper.
- `logs/`: detailed experiment log.
- `reports/`: result JSON files, CSV summaries, final summary, and final paper.
- `scripts/`: dataset, evaluation, final-suite runner, and report scripts.
- `requirements.txt`: Python dependencies used by the experiment.

## Main Report

Read `reports/counterexample_rule_repair_paper.md` first after report generation.

## Large Artifacts

Adapters and checkpoints are stored at:

`/workspace/large_artifacts/counterexample_rule_repair/models/`

Do not include that directory when downloading only the small package.
