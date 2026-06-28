# Qwen3.5-4B Operator Inventory Scaling Stress

Standalone scaling stress test for typed operator inventory search.

The experiment expands a same-signature operator library from single digits to hundreds of `list[int] -> int` operators, then compares exhaustive operator-hole search across one-hole and two-hole program templates. The goal is to find where full inventory enumeration remains cheap and identifiable, and where a Qwen3.5-4B inventory-conditioned top-k shortlister would become necessary.

Large artifacts, if any are produced later, belong outside this directory:

`/workspace/large_artifacts/qwen35_4b_operator_inventory_scaling_stress`

## Layout

- `configs/`: experiment configuration.
- `data/`: generated scaling benchmark records.
- `logs/`: chronological experiment log.
- `reports/`: result JSON, CSVs, figures, and final report.
- `run_logs/`: captured command output.
- `scripts/`: dataset, evaluation, and reporting entry points.
- `src/`: standalone operator library, task generator, and vectorized search.

## Main Commands

```bash
python scripts/build_dataset.py
python scripts/eval_scaling.py --data data/operator_scaling_eval.jsonl --output reports/operator_scaling_results.json
python scripts/make_report.py
```

