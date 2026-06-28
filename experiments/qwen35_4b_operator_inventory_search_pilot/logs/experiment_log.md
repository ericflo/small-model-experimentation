# Experiment Log

## 2026-06-24

Initialized a standalone no-training operator inventory search pilot.

Design commitments:

- Test type-colliding aggregate operators with identical signature `list[int] -> int`.
- Run the cheap pilot before any QLoRA training.
- Compare:
  - `arm1_closed_vocab`: current closed operator set, `sum`, `first`, `last`.
  - `arm0_full_inventory`: full in-context operator inventory, `sum`, `first`, `last`, `max`, `min`, `prod`, `gcd`.
- Keep downstream verified completion and active query logic unchanged in spirit: enumerate candidates, filter by visible execution, optionally request active query cases, score hidden cases.
- Track search cost via raw candidate counts and visible-consistent candidate counts.
- Store any future large model artifacts outside the experiment directory.

Initial implementation:

- Created standalone experiment package.
- Added local executor with `sum`, `first`, `last`, `max`, `min`, `prod`, and `gcd`.
- Added benchmark generator with three templates:
  - `mod_format`
  - `offset_format`
  - `threshold_gate`
- Added operator-hole completion and active query evaluator.
- Added README, config, and large-artifact manifest.

## Smoke Validation

Commands:

```bash
python -m py_compile scripts/*.py src/*.py
python scripts/build_dataset.py --records-per-family 2 > run_logs/dataset_smoke_console.log 2>&1
python scripts/eval_operator_search.py \
  --data data/operator_inventory_eval.jsonl \
  --output reports/_smoke_operator_search_results.json \
  --max-records 8 \
  > run_logs/eval_smoke_console.log 2>&1
```

Result:

- Syntax check passed.
- Smoke dataset contained 42 records.
- On the first 8 records, `arm0_full_inventory` recovered the target for all 5 held-out records; `arm1_closed_vocab` recovered 0 of 5 held-out records.
- This matched the intended falsification shape, so I proceeded to the full no-training pilot.

## Full No-Training Pilot

Commands:

```bash
python scripts/build_dataset.py > run_logs/dataset_build_console.log 2>&1
python scripts/eval_operator_search.py \
  --data data/operator_inventory_eval.jsonl \
  --output reports/operator_search_results.json \
  > run_logs/eval_operator_search_console.log 2>&1
python scripts/make_report.py > run_logs/report_generation_console.log 2>&1
```

Dataset:

- 210 total records.
- 90 in-bank operator records: `sum`, `first`, `last`.
- 120 held-out operator records: `max`, `min`, `prod`, `gcd`.
- All aggregate operators share signature `list[int] -> int`.
- Each record has 6 visible cases, 18 hidden cases, and 48 active-query pool cases.
- Templates: `mod_format`, `offset_format`, `threshold_gate`.

Primary result by operator status:

| arm | status | records | target raw | target visible | oracle hidden-all | selected hidden-all | avg raw candidates | avg visible-consistent operators |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `arm0_full_inventory` | held-out | 120 | 100.0% | 100.0% | 100.0% | 92.5% | 56.0 | 1.20 |
| `arm0_full_inventory` | in-bank | 90 | 100.0% | 100.0% | 100.0% | 90.0% | 56.0 | 1.32 |
| `arm1_closed_vocab` | held-out | 120 | 0.0% | 0.0% | 0.0% | 0.0% | 24.0 | 0.09 |
| `arm1_closed_vocab` | in-bank | 90 | 100.0% | 100.0% | 100.0% | 93.3% | 24.0 | 1.11 |

Held-out operator breakdown for `arm0_full_inventory`:

| operator | records | target raw | selected hidden-all | avg visible-consistent operators |
| --- | ---: | ---: | ---: | ---: |
| `gcd` | 30 | 100.0% | 76.7% | 1.37 |
| `max` | 30 | 100.0% | 100.0% | 1.10 |
| `min` | 30 | 100.0% | 96.7% | 1.30 |
| `prod` | 30 | 100.0% | 96.7% | 1.03 |

Active-query result on held-out operators:

| arm | policy | budget 0 | budget 1 | budget 2 | budget 3 |
| --- | --- | ---: | ---: | ---: | ---: |
| `arm0_full_inventory` | max-split | 92.5% | 99.2% | 100.0% | 100.0% |
| `arm0_full_inventory` | oracle-elimination | 92.5% | 100.0% | 100.0% | 100.0% |
| `arm1_closed_vocab` | max-split | 0.0% | 0.0% | 0.0% | 0.0% |
| `arm1_closed_vocab` | oracle-elimination | 0.0% | 0.0% | 0.0% | 0.0% |

Interpretation:

- The held-out operator target is always present once the inventory is allowed into the operator hole.
- Six visible cases underdetermine a small fraction of records, especially `gcd`, but operator-level active querying closes that ambiguity.
- At this scale, this is a bank/search-side fix rather than a training-side fix. A trained inventory-conditioned sketcher should be treated as a later top-k shortlisting mechanism for larger libraries, not as the immediate way to recover missing coverage.

Generated artifacts:

- Report: `reports/qwen35_4b_operator_inventory_search_pilot_report.md`
- Full results: `reports/operator_search_results.json`
- CSV summaries: `reports/status_summary.csv`, `reports/operator_summary.csv`, `reports/template_summary.csv`, `reports/active_summary.csv`
- Figures: `reports/figures/*.png`

## Final Audit

Commands/checks:

```bash
python -m py_compile scripts/*.py src/*.py
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f -size +50M -print
du -sh . /workspace/large_artifacts/qwen35_4b_operator_inventory_search_pilot
```

Audit result:

- Final syntax check passed.
- No Python cache directories remain.
- No file larger than 50 MB is present in the experiment directory.
- Experiment directory size: 4.4 MB.
- External large-artifact directory size: 0.
- A dynamic text scan against sibling experiment directory names found no references.
- No references to external experiment paths were found.
- PNG figures were opened and verified with PIL.
