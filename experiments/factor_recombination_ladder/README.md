# Factor Recombination Ladder

Standalone experiment package for testing whether trace-conditioned repair learns reusable factor recombination, rather than only learning seen rule templates.

## Result

The experiment found a sharp split between in-distribution repair learning and held-out factor recombination.

- Correct trace training worked well on seen factor combinations: 80.6% seen-IID repair and 63.9% format-shift repair.
- Factor labels improved format shift to 72.2%, but did not improve recombination holdout.
- Recombination holdout stayed low across trained conditions: trace 8.3%, labelled trace 8.3%, no-trace 10.0%, shuffled-trace 8.3%.
- Prompt ablations showed the trace content is behaviorally important: removing or shuffling traces collapsed seen/format performance for trace-trained adapters.
- Main readout: this ladder supports trace-conditioned template/mechanism learning, but not robust transfer to held-out factor-pair recombination.

Primary report:

- `reports/factor_recombination_ladder_report.md`

## Layout

- `configs/experiment.json`: fixed design, model, and hyperparameter settings.
- `data/`: generated JSONL datasets and dataset manifest.
- `scripts/`: dataset generation, LoRA training runner, final evaluation runner, and report generator.
- `reports/`: training manifests, final evaluation JSON, CSV summaries, and markdown report.
- `figures/`: generated plots.
- `logs/experiment_log.md`: chronological run log.

Large model artifacts are intentionally outside this directory:

- `large_artifacts/factor_recombination_ladder/models/`

The compact experiment directory contains scripts, logs, data, reports, and figures only. Adapter weights and training checkpoints are separated so this directory can be downloaded without multi-GB model files.

## Core Commands

Build data:

```bash
python experiments/factor_recombination_ladder/scripts/build_ladder_dataset.py \
  --output-dir experiments/factor_recombination_ladder/data
```

Train adapters:

```bash
python experiments/factor_recombination_ladder/scripts/run_training.py --suite all
```

Run final evaluations:

```bash
python experiments/factor_recombination_ladder/scripts/run_final_evaluations.py --suite all
```

Generate report:

```bash
python experiments/factor_recombination_ladder/scripts/make_report.py
```
