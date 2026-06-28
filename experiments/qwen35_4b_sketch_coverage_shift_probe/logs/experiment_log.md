# Experiment Log

## 2026-06-24 08:20 UTC

Initialized a standalone coverage-shift probe.

Design commitments:

- Use a local executable substrate with typed sketches and verified completion.
- Do not train a new Qwen adapter for the first pass.
- Keep the completion bank intentionally unchanged while extending only the executor to score shifted primitives.
- Compare three sketch conditions:
  - `auto`: the existing target-sketch generator.
  - `manual`: a hand-typed sketch with the intended operator shape and typed holes.
  - `erased`: a low-information sketch that preserves only output format or branch labels.
- Compare three task regimes:
  - `control_in_bank`: operations and names the completion bank is tuned for.
  - `name_shift`: same operations but renamed variables and constants.
  - `primitive_shift`: executor-visible primitives that the completion bank was not tuned to propose.
- Evaluate candidate coverage first; only interpret active-query selection where coverage exists.

Initial implementation:

- Created standalone experiment directory.
- Copied local executor and sketch completion mechanics into `src/`.
- Extended the executor with `mul`, `abs`, `max`, `min`, `prod`, and `startswith`.
- Added shifted task definitions with visible, hidden, and query-pool cases.
- Added dataset builder, README, config, and large-artifact manifest.

Validation:

- `python -m py_compile scripts/build_dataset.py src/*.py`
- Result: passed.

## 2026-06-24 08:31 UTC

Built the first shifted dataset with 12 records per family.

Command:

```bash
python scripts/build_dataset.py > run_logs/dataset_build_console.log 2>&1
```

Observed:

- Total records: `144`.
- `control_in_bank`: `36`.
- `name_shift`: `36`.
- `primitive_shift`: `72`.
- Visible cases per record: `6`.
- Hidden cases per record: `18`.
- Query-pool cases per record: `48`.

Sketch inspection showed the intended contrast:

- Control and name-shift `auto` sketches were typed correctly.
- Shifted primitives such as `max`, `min`, `prod`, and `startswith` were mistyped by `auto` sketches in most families.
- `manual` sketches preserved the shifted operator shape and typed holes.
- `erased` sketches preserved only output format or branch labels.

## 2026-06-24 08:38 UTC

Ran evaluator smoke test.

Command:

```bash
python scripts/run_coverage_probe.py --data data/shifted_coverage_eval.jsonl --output reports/_smoke_coverage_probe.json --max-records 3 --hole-options 8,16 --active-hole-options 16 --max-programs-per-sketch 500 > run_logs/coverage_smoke_console.log 2>&1
```

Observed:

- Smoke completed successfully.
- Control target coverage was present.
- Primitive `auto`/`erased` failures appeared in the smoke slice.
- Primitive `manual` sketches recovered targets in the smoke slice.
- Active-query rows were emitted for both max-split and oracle-elimination policies.

Decision:

- Proceed to the full grid, but watch runtime because high-arity sketches can enumerate many completions.

## 2026-06-24 08:44 UTC

Started a full grid with a 4000-program completion cap, then stopped it after the first minute because high-arity rows were too slow for the information gain.

Stopped command:

```bash
python scripts/run_coverage_probe.py --data data/shifted_coverage_eval.jsonl --output reports/coverage_probe.json --hole-options 8,16,28 --active-hole-options 28 --max-programs-per-sketch 4000 > run_logs/coverage_full_console.log 2>&1
```

Reason:

- Early tuple-style rows were taking tens of seconds each.
- The smoke run had already recovered the relevant high-arity targets within 500 completions.
- The falsification question is coverage under shift, not exhaustive enumeration at a very large cap.

Adjustment:

- Rebuilt the dataset at 8 records per family to preserve family balance while reducing runtime.
- Used `1000` completions per sketch for the full run.

Command:

```bash
python scripts/build_dataset.py --records-per-family 8 > run_logs/dataset_build_records8_console.log 2>&1
```

Observed:

- Total records: `96`.
- `control_in_bank`: `24`.
- `name_shift`: `24`.
- `primitive_shift`: `48`.

## 2026-06-24 08:55 UTC

Completed the full coverage and active-query grid.

Command:

```bash
python scripts/run_coverage_probe.py --data data/shifted_coverage_eval.jsonl --output reports/coverage_probe.json --hole-options 8,16,28 --active-hole-options 28 --max-programs-per-sketch 1000 > run_logs/coverage_full_cap1000_console.log 2>&1
```

Runtime:

- `864` coverage jobs completed in `9:35`.

Primary cap-28 coverage results:

- `control_in_bank`, `auto`: target coverage `24/24`, candidate oracle `24/24`, visible-selected hidden all-pass `16/24`.
- `control_in_bank`, `manual`: target coverage `24/24`, candidate oracle `24/24`, visible-selected hidden all-pass `16/24`.
- `control_in_bank`, `erased`: target coverage `8/24`, candidate oracle `8/24`, visible-selected hidden all-pass `8/24`.
- `name_shift`, `auto`: target coverage `24/24`, candidate oracle `24/24`, visible-selected hidden all-pass `12/24`.
- `name_shift`, `manual`: target coverage `24/24`, candidate oracle `24/24`, visible-selected hidden all-pass `12/24`.
- `name_shift`, `erased`: target coverage `0/24`, candidate oracle `1/24`, visible-selected hidden all-pass `1/24`.
- `primitive_shift`, `auto`: target coverage `8/48`, candidate oracle `8/48`, visible-selected hidden all-pass `8/48`.
- `primitive_shift`, `manual`: target coverage `48/48`, candidate oracle `48/48`, visible-selected hidden all-pass `44/48`.
- `primitive_shift`, `erased`: target coverage `0/48`, candidate oracle `0/48`, visible-selected hidden all-pass `0/48`.

Family-level primitive-shift cap-28 target coverage:

- `primitive_mul_sum`: `auto 8/8`, `manual 8/8`, `erased 0/8`.
- `primitive_abs_min_delta`: `auto 0/8`, `manual 8/8`, `erased 0/8`.
- `primitive_max_gate`: `auto 0/8`, `manual 8/8`, `erased 0/8`.
- `primitive_max_mod`: `auto 0/8`, `manual 8/8`, `erased 0/8`.
- `primitive_prefix_gate`: `auto 0/8`, `manual 8/8`, `erased 0/8`.
- `primitive_prod_mod`: `auto 0/8`, `manual 8/8`, `erased 0/8`.

Active-query cap-28 diagnostic:

- Control `auto`/`manual`: visible-selected `16/24`; active max-split reached `21/24` at budget 3; oracle elimination reached `24/24`.
- Name-shift `auto`/`manual`: visible-selected `12/24`; active max-split reached `22/24` at budget 3; oracle elimination reached `24/24` by budget 2.
- Primitive-shift `manual`: visible-selected `44/48`; active max-split reached `46/48` at budget 3; oracle elimination reached `48/48` by budget 2.
- Primitive-shift `auto`: stayed at `8/48` for every active budget because coverage was absent for five of six families.
- Primitive-shift `erased`: stayed at `0/48`.

Interpretation:

- Completion coverage survives name shift when the sketch carries the correct structure.
- Completion coverage does not survive primitive shift when the sketcher mistypes or omits the shifted operator.
- If the shifted operator is explicitly present in a correctly typed sketch, the completion bank can usually fill the arguments and recover target coverage.
- Active querying is useful for disambiguating survivors after coverage exists, but it cannot recover absent programs.

## 2026-06-24 09:08 UTC

Generated report, CSVs, and figures.

Command:

```bash
python scripts/make_report.py > run_logs/report_generation_v2_console.log 2>&1
```

Artifacts:

- Final report: `reports/qwen35_4b_sketch_coverage_shift_probe_report.md`.
- Full result JSON: `reports/coverage_probe.json`.
- CSVs:
  - `reports/coverage_by_shift.csv`
  - `reports/coverage_by_family.csv`
  - `reports/active_by_shift.csv`
- Figures:
  - `reports/figures/target_coverage_by_shift.png`
  - `reports/figures/oracle_hidden_by_shift.png`
  - `reports/figures/selected_hidden_by_shift.png`
  - `reports/figures/cap_sensitivity.png`
  - `reports/figures/family_target_coverage_heatmap.png`
  - `reports/figures/active_query_diagnostic.png`

## 2026-06-24 09:14 UTC

Final packaging and audit completed.

Checks:

- `python -m py_compile scripts/*.py src/*.py` passed before cache cleanup.
- Removed generated `__pycache__` directories.
- Verified no stale package-name references in the standalone experiment tree.
- Verified no file larger than `50M` under `/workspace/experiments/qwen35_4b_sketch_coverage_shift_probe`.
- Verified experiment package size: `4.1M`.
- Verified large artifact directory size: `0`.
- Verified all 6 generated PNG figures are readable and non-empty.

Final locations:

- Downloadable experiment package: `/workspace/experiments/qwen35_4b_sketch_coverage_shift_probe`.
- Large artifact root: `/workspace/large_artifacts/qwen35_4b_sketch_coverage_shift_probe`.
- Final report: `reports/qwen35_4b_sketch_coverage_shift_probe_report.md`.
