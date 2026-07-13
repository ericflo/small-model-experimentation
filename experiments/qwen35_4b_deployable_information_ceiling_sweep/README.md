# Qwen3.5-4B Deployable Information Ceiling Sweep

**Status:** finished

This standalone diagnostic asks whether the hard low-information regime is limited by deployable information or by a trainable probe-selection policy.

There is no model training in this package. The deployable policy is greedy max expected information gain under a uniform posterior over all verifier-surviving candidates. The comparison policy is a target-aware oracle and is only a headroom measurement.

Main outputs:

- `reports/qwen35_4b_deployable_information_ceiling_sweep_report.md`
- `reports/figures/`
- `reports/*.csv`
- `reports/eval/*.json`
- `logs/experiment_log.md`

Reproduction:

```bash
python scripts/eval_information_sweep.py --max-budget 10 --visible-extra 0 4 8 12
python scripts/make_report.py
```
