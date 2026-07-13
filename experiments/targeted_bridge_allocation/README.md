# Targeted Bridge Allocation

**Status:** finished

Standalone experiment package for testing whether fixed-budget targeted bridge allocation improves trace-conditioned recombination repair compared with uniform bridge allocation.

## Design

Every trained condition has exactly 240 training records. Exact bridge examples are drawn from five held-out recombination families. Adding bridge examples removes seen-combination examples, so allocation strategy is the tested variable rather than total dataset size.

Core allocation conditions:

- `uniform2_trace`: 2 exact bridge examples for each held-out family.
- `uniform4_trace`: 4 exact bridge examples for each held-out family.
- `hard_target_trace`: 8 bridge examples for each target family and 2 for each responsive-control family.
- `hard_target_seen_preserving_trace`: 6 bridge examples for each target family and 1 for each responsive-control family, matching the uniform4 bridge total.
- `easy_target_control_trace`: same bridge total as `hard_target_trace`, concentrated on responsive-control families.
- `modulo16_trace`, `length16_trace`, `tuple16_trace`: light single-family high-dose probes, with one target family raised to 16 bridge examples and all other held-out families kept at 2.
- `hard_target_no_trace` and `hard_target_shuffled_trace`: controls for whether the main targeted allocation depends on aligned trace evidence.

## Layout

- `configs/experiment.json`: fixed design, model, training, and evaluation settings.
- `data/`: generated JSONL datasets and dataset manifest.
- `scripts/`: dataset generation, LoRA training runner, final evaluation runner, evaluator, and report generator.
- `reports/`: training manifests, final evaluation JSON, CSV summaries, and markdown report.
- `figures/`: generated plots.
- `logs/experiment_log.md`: chronological run log.
- `run_logs/`: raw console logs for dataset build, training, evaluation, and report generation.

Large model artifacts are intentionally outside this directory:

- `large_artifacts/targeted_bridge_allocation/models/`

The compact experiment directory should contain scripts, logs, data, reports, and figures only. Adapter weights and training checkpoints are separated so this directory can be downloaded without multi-GB model files.

## Final Result Snapshot

The full evaluation matrix completed 39/39 jobs. Main recombination-holdout results:

| Condition | Recombination Holdout |
| --- | --- |
| `frozen_trace` | 0.0% (0/60) |
| `uniform2_trace` | 28.3% (17/60) |
| `uniform4_trace` | 25.0% (15/60) |
| `hard_target_trace` | 33.3% (20/60) |
| `hard_target_seen_preserving_trace` | 25.0% (15/60) |
| `easy_target_control_trace` | 31.7% (19/60) |
| `modulo16_trace` | 33.3% (20/60) |
| `length16_trace` | 20.0% (12/60) |
| `tuple16_trace` | 28.3% (17/60) |
| `hard_target_no_trace` | 8.3% (5/60) |
| `hard_target_shuffled_trace` | 10.0% (6/60) |

Key readout: `hard_target_trace` tied the top aggregate holdout score while covering more target-family successes than the easy-target control. Trace quality was decisive: no-trace and shuffled-trace controls collapsed on the target families.

Primary outputs:

- `reports/targeted_bridge_allocation_report.md`
- `reports/final_results.csv`
- `reports/final_results_by_family.csv`
- `reports/final_results_by_factor.csv`
- `figures/final_repair_by_condition_split.png`
- `figures/recombination_holdout_by_family.png`
- `reports/large_artifacts_manifest.md`

## Core Commands

Build data:

```bash
python experiments/targeted_bridge_allocation/scripts/build_allocation_dataset.py \
  --output-dir experiments/targeted_bridge_allocation/data
```

Train adapters:

```bash
python experiments/targeted_bridge_allocation/scripts/run_training.py --suite all
```

Run final evaluations:

```bash
python experiments/targeted_bridge_allocation/scripts/run_final_evaluations.py --suite all
```

Generate report:

```bash
python experiments/targeted_bridge_allocation/scripts/make_report.py
```
