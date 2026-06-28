# Qwen3.5-4B Sketch Coverage Shift Probe

Standalone falsification experiment for typed-sketch verified completion under task shift.

The experiment does not train a new adapter. It tests the load-bearing coverage assumption behind typed-sketch synthesis: when the task substrate changes, does the correct executable completion still appear in the bounded candidate set?

Large artifacts, if any are produced later, belong outside this directory:

`/workspace/large_artifacts/qwen35_4b_sketch_coverage_shift_probe`

## Layout

- `configs/`: experiment configuration.
- `data/`: generated shifted task records and dataset manifest.
- `logs/`: chronological experiment log.
- `reports/`: metrics, CSVs, figures, and final writeup.
- `run_logs/`: captured command output.
- `scripts/`: dataset, evaluation, and reporting entry points.
- `src/`: standalone executor, typed-sketch completer, and shifted task definitions.

## Main Commands

```bash
python scripts/build_dataset.py
python scripts/run_coverage_probe.py --data data/shifted_coverage_eval.jsonl --output reports/coverage_probe.json
python scripts/make_report.py
```

