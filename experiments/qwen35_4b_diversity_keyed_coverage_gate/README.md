# Qwen3.5-4B Diversity-Keyed Coverage Gate

This top-level README was generated during repository normalization because the imported experiment did not include one.

- Source track: `track-z`
- Primary report: [reports/qwen35_4b_diversity_keyed_coverage_gate_report.md](reports/qwen35_4b_diversity_keyed_coverage_gate_report.md)
- Metadata: [metadata.yaml](metadata.yaml)

## How To Read

Start with the primary report, then inspect `data/`, `reports/`, `analysis/`, `src/`, and `scripts/` as available. This folder remains self-contained; do not move its run data into shared directories.

## Summary

This experiment tests whether held-out MBPP tasks missed by a small direct sample pool are diversity-limited or capability-limited. The practical question is whether a small posttraining objective should try to reshape the model into a better ensemble sampler, or whether inference-time diverse sampling already captures the available headroom.
