# real_transform_abi_gate_with_counterexamples

**Status:** finished

Standalone no-training experiment for deterministic transformation ABI coverage with counterexample filtering.

The package defines two independently curated transformation domains:

- clean CSV/ETL-style row transformations
- irregular date/ID/string normalization transformations

It freezes a generic ABI before evaluation, measures raw coverage on visible plus standard hidden tests, then applies adversarial counterexample tests generated from the task reference semantics. No model training or checkpoints are produced.

## Reproduce

```bash
python scripts/run_gate.py --data-dir data --reports-dir reports
python scripts/make_report.py
```

The main writeup is `reports/report.md`; figures are in `reports/figures/`.
