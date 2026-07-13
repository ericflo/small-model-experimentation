# Qwen3.5-4B Operator Inventory Search Pilot

**Status:** finished

Standalone no-training pilot for open-vocabulary operator identification.

The experiment tests whether type-colliding held-out aggregate operators can be recovered by operator-level inventory search before training an inventory-conditioned Qwen3.5-4B sketcher. Every aggregate candidate has signature `list[int] -> int`, so type alone cannot identify the operator.

Large artifacts, if any are produced later, belong outside this directory:

`/workspace/large_artifacts/qwen35_4b_operator_inventory_search_pilot`

## Layout

- `configs/`: experiment configuration.
- `data/`: generated operator-collision task records.
- `logs/`: chronological experiment log.
- `reports/`: result JSON, CSVs, figures, and final report.
- `run_logs/`: captured command output.
- `scripts/`: dataset, evaluation, and reporting entry points.
- `src/`: standalone executor, task generator, and operator-hole search.

## Main Commands

```bash
python scripts/build_dataset.py
python scripts/eval_operator_search.py --data data/operator_inventory_eval.jsonl --output reports/operator_search_results.json
python scripts/make_report.py
```

