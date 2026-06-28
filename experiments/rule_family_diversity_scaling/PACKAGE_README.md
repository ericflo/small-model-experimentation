# Rule-Family Diversity Scaling Package

This package is self-contained except for trained LoRA adapters and checkpoints, which are intentionally stored outside the downloadable directory.

Final compact outputs:

- Paper: `reports/rule_family_diversity_scaling_paper.md`
- Summary: `reports/rule_family_diversity_scaling_summary.md`
- Core results CSV: `reports/final_core_results.csv`
- Ablation results CSV: `reports/final_ablation_results.csv`
- Per-family results CSV: `reports/final_trace_by_family.csv`
- Full per-record JSON outputs: `reports/final/`

Large artifacts:

`/workspace/large_artifacts/rule_family_diversity_scaling/`

Rebuild the dataset with:

```bash
python experiments/rule_family_diversity_scaling/scripts/build_diversity_dataset.py --output-dir experiments/rule_family_diversity_scaling/data
```

Run final evaluations with:

```bash
python experiments/rule_family_diversity_scaling/scripts/run_final_evaluations.py --suite all --max-new-tokens 256
```

Generate the report with:

```bash
python experiments/rule_family_diversity_scaling/scripts/make_report.py
```
