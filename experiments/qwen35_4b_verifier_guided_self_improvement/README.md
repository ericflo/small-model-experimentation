# Qwen3.5-4B Verifier-Guided Self-Improvement Report

This top-level README was generated during repository normalization because the imported experiment did not include one.

- Source track: `track-z`
- Primary report: [reports/qwen35_4b_verifier_guided_self_improvement_report.md](reports/qwen35_4b_verifier_guided_self_improvement_report.md)
- Metadata: [metadata.yaml](metadata.yaml)

## How To Read

Start with the primary report, then inspect `data/`, `reports/`, `analysis/`, `src/`, and `scripts/` as available. This folder remains self-contained; do not move its run data into shared directories.

## Summary

The main result is negative for the central question. Verified self-training did not raise held-out generation coverage under this local LoRA/data budget. The 20-task smoke signal was positive, but the 150-task held-out run regressed slightly.
