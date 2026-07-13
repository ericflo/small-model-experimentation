# Qwen3.5-4B Trained vs Frozen Repair MDP Report

**Status:** finished

This top-level README was generated during repository normalization because the imported experiment did not include one.

- Source track: `track-z`
- Primary report: [reports/qwen35_4b_trained_vs_frozen_repair_mdp_report.md](reports/qwen35_4b_trained_vs_frozen_repair_mdp_report.md)
- Metadata: [metadata.yaml](metadata.yaml)

## How To Read

Start with the primary report, then inspect `data/`, `reports/`, `analysis/`, `src/`, and `scripts/` as available. This folder remains self-contained; do not move its run data into shared directories.

## Summary

This experiment tested whether a trained repair policy can expand held-out coding coverage beyond frozen Qwen self-repair, under a fair comparison against spending the same estimated model-forward-token budget on more direct samples.
