# Bridge-Dose Recombination Curriculum

Standalone experiment package for testing whether a small number of exact factor-pair bridge examples is enough to unlock trace-conditioned recombination generalization.

## Final Result Snapshot

Final evaluation completed for all 39 planned jobs across seen IID, format-shift, and recombination-holdout splits.

| Condition | Seen IID | Format Shift | Recombination Holdout |
| --- | ---: | ---: | ---: |
| Frozen trace | 0.0% (0/36) | 0.0% (0/36) | 0.0% (0/60) |
| Dose 0 trace | 77.8% (28/36) | 80.6% (29/36) | 6.7% (4/60) |
| Dose 1 trace | 86.1% (31/36) | 80.6% (29/36) | 15.0% (9/60) |
| Dose 2 trace | 83.3% (30/36) | 75.0% (27/36) | 28.3% (17/60) |
| Dose 4 trace | 72.2% (26/36) | 77.8% (28/36) | 31.7% (19/60) |
| Dose 8 trace | 58.3% (21/36) | 80.6% (29/36) | 30.0% (18/60) |
| Near-miss focus trace | 83.3% (30/36) | 80.6% (29/36) | 8.3% (5/60) |

Core readout:

- Exact bridge examples caused a large recombination jump: dose 0 to dose 8 improved holdout repair from 6.7% to 30.0%.
- Dose 4 was the best absolute holdout condition at 31.7%, but dose 2 captured most of the gain while preserving stronger seen-IID behavior.
- Near-miss focus did not substitute for exact bridge pairs: 8.3% holdout despite strong seen/format performance.
- Trace alignment was necessary. Dose 8 no-trace and shuffled-trace controls stayed at 8.3% holdout, and prompt-time trace ablations collapsed the trained dose 8 trace adapter.

The full report is in `reports/bridge_dose_recombination_curriculum_report.md`.

## Layout

- `configs/experiment.json`: fixed design, model, training, and evaluation settings.
- `data/`: generated JSONL datasets and dataset manifest.
- `scripts/`: dataset generation, LoRA training runner, final evaluation runner, evaluator, and report generator.
- `reports/`: training manifests, final evaluation JSON, CSV summaries, and markdown report.
- `figures/`: generated plots.
- `logs/experiment_log.md`: chronological run log.
- `run_logs/`: raw console logs for dataset build, training, evaluation, and report generation.

Large model artifacts are intentionally outside this directory:

- `large_artifacts/bridge_dose_recombination_curriculum/models/`

The compact experiment directory should contain scripts, logs, data, reports, and figures only. Adapter weights and training checkpoints are separated so this directory can be downloaded without multi-GB model files.

## Key Outputs

- `reports/bridge_dose_recombination_curriculum_report.md`: final narrative report.
- `reports/final_results.csv`: one-row-per-condition/split evaluation summary.
- `reports/final_results_by_family.csv`: recombination holdout broken down by synthetic family.
- `reports/final_results_by_factor.csv`: recombination holdout broken down by primitive factor.
- `reports/final/final_evaluation_jobs.json`: complete final evaluation job manifest.
- `reports/training/training_jobs.json`: complete training job manifest.
- `figures/final_repair_by_condition_split.png`: main split-level repair chart.
- `figures/recombination_holdout_by_family.png`: family-level holdout chart.
- `logs/experiment_log.md`: chronological run log with decisions, commands, and observations.

## Download Notes

Download `experiments/bridge_dose_recombination_curriculum/` for the complete compact experiment package. It contains data, code, logs, reports, figures, and run manifests.

Do not include `large_artifacts/bridge_dose_recombination_curriculum/` unless adapter weights/checkpoints are needed. That directory contains the multi-GB LoRA outputs and checkpoint files.

## Core Commands

Build data:

```bash
python experiments/bridge_dose_recombination_curriculum/scripts/build_bridge_dataset.py \
  --output-dir experiments/bridge_dose_recombination_curriculum/data
```

Train adapters:

```bash
python experiments/bridge_dose_recombination_curriculum/scripts/run_training.py --suite all
```

Run final evaluations:

```bash
python experiments/bridge_dose_recombination_curriculum/scripts/run_final_evaluations.py --suite all
```

Generate report:

```bash
python experiments/bridge_dose_recombination_curriculum/scripts/make_report.py
```
