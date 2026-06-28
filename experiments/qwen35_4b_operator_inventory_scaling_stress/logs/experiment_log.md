# Experiment Log

## 2026-06-24

Initialized a standalone operator inventory scaling stress test.

Design commitments:

- Generate a large library of same-signature `list[int] -> int` operators.
- Sweep inventory sizes from 8 through 512 operators.
- Include both one-hole and two-hole templates so candidate count scales as `N` and `N^2`.
- Keep the first run no-training: measure exhaustive search cost, coverage, visible ambiguity, active-query lift, and fixed-budget prefix coverage.
- Store any future model artifacts outside the experiment directory.

Initial implementation:

- Added a standalone library generator with 512 same-signature operators across core, order-statistic, count, modular, and bounded aggregate families.
- Added four templates:
  - `single_mod`
  - `single_offset`
  - `pair_affine_mod`
  - `pair_compare_gate`
- Added vectorized exhaustive search over one-hole and two-hole candidate ids.
- Added active query evaluation for max-split and oracle-elimination policies.
- Added reporting with CSV summaries and PNG figures.

## Smoke Validation

Commands:

```bash
python -m py_compile scripts/*.py src/*.py
python scripts/build_dataset.py --library-sizes 8,16 --records-per-template 2 > run_logs/dataset_smoke_console.log 2>&1
python scripts/eval_scaling.py \
  --data data/operator_scaling_eval.jsonl \
  --output reports/_smoke_operator_scaling_results.json \
  > run_logs/eval_smoke_console.log 2>&1
```

Result:

- Syntax check passed.
- Smoke dataset contained 16 records.
- Candidate rows: 16.
- Active rows: 128.
- Target-visible coverage was 100% for both one-hole and two-hole records.
- Two-hole ambiguity appeared immediately: average visible-consistent candidates rose from 1.5 at 8 operators to 11.75 at 16 operators.

## Full Scaling Sweep

Commands:

```bash
python scripts/build_dataset.py > run_logs/dataset_build_console.log 2>&1
python scripts/eval_scaling.py \
  --data data/operator_scaling_eval.jsonl \
  --output reports/operator_scaling_results.json \
  > run_logs/eval_scaling_console.log 2>&1
python scripts/make_report.py > run_logs/report_generation_console.log 2>&1
```

Dataset:

- 336 total records.
- Library sizes: 8, 16, 32, 64, 128, 256, 512.
- 48 records per library size.
- 24 one-hole records and 24 two-hole records per library size.
- 6 visible cases, 18 hidden cases, and 48 query-pool cases per record.
- 512 generated same-signature operators.

Primary result by library size and hole count:

| library | holes | records | raw candidates | target visible | oracle hidden-all | selected hidden-all | visible candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 24 | 8 | 100.0% | 100.0% | 100.0% | 1.00 |
| 8 | 2 | 24 | 64 | 100.0% | 100.0% | 75.0% | 2.42 |
| 64 | 1 | 24 | 64 | 100.0% | 100.0% | 87.5% | 1.96 |
| 64 | 2 | 24 | 4096 | 100.0% | 100.0% | 45.8% | 300.79 |
| 512 | 1 | 24 | 512 | 100.0% | 100.0% | 100.0% | 26.54 |
| 512 | 2 | 24 | 262144 | 100.0% | 100.0% | 45.8% | 8695.79 |

Two-hole template breakdown:

| library | template | selected hidden-all | visible candidates |
| --- | --- | ---: | ---: |
| 128 | `pair_affine_mod` | 75.0% | 2.83 |
| 128 | `pair_compare_gate` | 0.0% | 3729.92 |
| 512 | `pair_affine_mod` | 66.7% | 357.50 |
| 512 | `pair_compare_gate` | 25.0% | 17034.08 |

Active-query result on two-hole records:

| library | policy | budget 0 | budget 1 | budget 2 | budget 3 |
| --- | --- | ---: | ---: | ---: | ---: |
| 64 | max-split | 45.8% | 50.0% | 70.8% | 83.3% |
| 64 | oracle-elimination | 45.8% | 70.8% | 87.5% | 95.8% |
| 512 | max-split | 45.8% | 62.5% | 66.7% | 70.8% |
| 512 | oracle-elimination | 45.8% | 66.7% | 83.3% | 83.3% |

Fixed-prefix coverage on two-hole records:

| library | 1024 candidates | 4096 candidates | 16384 candidates |
| --- | ---: | ---: | ---: |
| 64 | 37.5% | 100.0% | 100.0% |
| 128 | 0.0% | 29.2% | 100.0% |
| 256 | 4.2% | 8.3% | 37.5% |
| 512 | 0.0% | 4.2% | 12.5% |

Interpretation:

- Exhaustive target reachability remains 100% through 512 operators because the target is in the inventory and full search enumerates all candidates.
- The real failure mode is the combination of quadratic candidate growth and residual visible ambiguity.
- `pair_compare_gate` is the stress case: binary outputs leave thousands of visible-consistent two-operator candidates at large library sizes.
- Active querying helps, but it operates after full enumeration and does not solve the candidate-budget problem.
- The next trained experiment should target Qwen3.5-4B inventory-conditioned top-k shortlisting for two-hole programs, measured by coverage at fixed budgets of 1024, 4096, and 16384 candidates.

Generated artifacts:

- Report: `reports/qwen35_4b_operator_inventory_scaling_stress_report.md`
- Full results: `reports/operator_scaling_results.json`
- CSV summaries: `reports/library_depth_summary.csv`, `reports/library_template_summary.csv`, `reports/target_bucket_summary.csv`, `reports/prefix_summary.csv`, `reports/active_summary.csv`
- Figures: `reports/figures/*.png`

## Final Audit

Commands/checks:

```bash
python -m py_compile scripts/*.py src/*.py
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f -size +50M -print
du -sh . /workspace/large_artifacts/qwen35_4b_operator_inventory_scaling_stress
```

Audit result:

- Final syntax check passed.
- No Python cache directories remain.
- No file larger than 50 MB is present in the experiment directory.
- Experiment directory size: 4.0 MB.
- External large-artifact directory size: 0.
- A dynamic text scan against sibling experiment directory names found no references.
- No standalone-forbidden temporal references were found.
- PNG figures were opened and verified with PIL.
