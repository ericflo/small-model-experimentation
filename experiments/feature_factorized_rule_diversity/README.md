# Feature-Factorized Rule Diversity

Standalone experiment package for testing whether trace-conditioned repair transfer is driven more by isolated primitive factor coverage, analogous multi-factor composition coverage, or a fixed-budget mixture of both.

## Layout

- `configs/experiment.json`: fixed design, model, and hyperparameter settings.
- `data/`: generated JSONL datasets and dataset manifest.
- `scripts/`: dataset generation, LoRA training runner, final evaluation runner, and report generator.
- `reports/`: training manifests, final evaluation JSON, CSV summaries, and markdown report.
- `figures/`: generated plots.
- `logs/experiment_log.md`: chronological run log.

## Final Report

- Main report: `reports/feature_factorized_rule_diversity_report.md`
- Detailed log: `logs/experiment_log.md`
- Final results CSV: `reports/final_results.csv`
- Final evaluation manifest: `reports/final/final_evaluation_jobs.json`

Key readout: aligned trace supervision is necessary, but fixed-budget mixed factor coverage did not improve broad recombination. The best recombination score was the composite-trace adapter at 14/60; mixed trace reached 13/60 and mostly transferred only the `sorted_join_holdout` family.

Large model artifacts are intentionally outside this directory:

- `large_artifacts/feature_factorized_rule_diversity/models/`

## Core Commands

Build data:

```bash
python experiments/feature_factorized_rule_diversity/scripts/build_factorized_dataset.py \
  --output-dir experiments/feature_factorized_rule_diversity/data
```

Train adapters:

```bash
python experiments/feature_factorized_rule_diversity/scripts/run_training.py --suite all
```

Run final evaluations:

```bash
python experiments/feature_factorized_rule_diversity/scripts/run_final_evaluations.py --suite all
```

Generate report:

```bash
python experiments/feature_factorized_rule_diversity/scripts/make_report.py
```
