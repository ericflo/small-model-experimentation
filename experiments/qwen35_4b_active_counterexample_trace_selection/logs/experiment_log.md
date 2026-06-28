# Experiment Log

## 2026-06-24 02:26 UTC

Initialized a standalone active counterexample trace-selection experiment package.

Design commitments:

- Use `Qwen/Qwen3.5-4B`.
- Train a fresh local sketch LoRA for this experiment instead of depending on another run's adapter.
- Keep checkpoints and adapters under `/workspace/large_artifacts/qwen35_4b_active_counterexample_trace_selection`.
- Store each eval record with its own deterministic active-query case pool.
- Compare original visible-trace selection against random extra cases, active max-split cases, and an oracle-elimination upper bound.

## 2026-06-24 02:34 UTC

Implemented:

- `scripts/build_dataset.py`: dataset builder with per-record active query pools.
- `scripts/eval_active_selection.py`: Qwen sketch generation, typed-sketch candidate synthesis, and active selection policies.
- `scripts/make_report.py`: CSV summaries, plots, and final Markdown report generation.

Next check: syntax/import validation, then a target-sketch smoke run before model training.

## 2026-06-24 02:36 UTC

Dataset build iteration 1 failed before writing data:

```text
TypeError: base_records() got an unexpected keyword argument 'case_pool_count'
```

Cause: the prior base-family record helper did not expose active query-pool generation.

Fix: patched local `src/data_gen.py` so `base_records` accepts `case_pool_count` and `case_mode`, passing both through to `make_record`.

## 2026-06-24 02:41 UTC

Target-sketch smoke iteration 1 was interrupted after ~74 seconds on record 1.

Cause: the evaluator called raw selection repeatedly, reparsing and reevaluating all programs against hidden cases for every policy snapshot.

Fixes:

- Added `ProgramBank` to parse each candidate program once per record.
- Cached program outputs by `(program_index, case_key)`.
- Replaced repeated raw selection with cached per-record selection.

## 2026-06-24 02:45 UTC

Target-sketch smoke iteration 2 still ran too slowly with a 384-case scoring pool.

Fix:

- Added deterministic `--max-query-pool-cases`.
- All policies now share the same capped per-record query pool.

## 2026-06-24 02:48 UTC

Target-sketch smoke iteration 3 completed:

- Command: `python scripts/eval_active_selection.py --data data/eval/dsl_eval_ceiling.jsonl --sketch-source target --output reports/eval/_smoke_target_ceiling10.json --max-records 10 --budgets 0,1,2,3,6 --random-repeats 2 --max-total-programs-per-record 4000 --max-policy-candidates 128 --max-query-pool-cases 48`
- Result: candidate oracle `10/10`; visible-only selected `10/10`.
- Interpretation: smoke validates evaluator mechanics; it is not diagnostic because this first 10-record slice is already solved from visible traces.

Next step: train the experiment-local Qwen3.5-4B sketch LoRA.

## 2026-06-24 03:07 UTC

Sketch LoRA training completed successfully.

Command:

```bash
python scripts/train_adapter.py --train data/static_bridge_80/dsl_train.jsonl --eval data/eval/dsl_eval_iid.jsonl --task sketch --target-field target_sketch --prompt-mode trace --output-dir /workspace/large_artifacts/qwen35_4b_active_counterexample_trace_selection/models/sketch_lora --epochs 2.0 --lr 1.5e-4 --rank 32 --alpha 64 --dropout 0.05 --grad-accum 8 --save-steps 30 --eval-steps 30
```

Observed:

- Trainable parameters: `42,467,328`.
- Step 30 eval loss: `0.0007196`.
- Step 60 eval loss: `0.0004834`.
- Train runtime: `959.9` seconds.
- Final adapter/checkpoint tree size: `709M`.
- Large artifact location: `/workspace/large_artifacts/qwen35_4b_active_counterexample_trace_selection/models/sketch_lora`.

Next step: model-generated smoke eval on the ceiling split.

## 2026-06-24 03:12 UTC

Model-generated ceiling smoke completed.

Command:

```bash
python scripts/eval_active_selection.py --data data/eval/dsl_eval_ceiling.jsonl --adapter /workspace/large_artifacts/qwen35_4b_active_counterexample_trace_selection/models/sketch_lora --sketch-source model --output reports/eval/_smoke_model_ceiling10.json --max-records 10 --budgets 0,1,2,3,6 --random-repeats 2 --num-samples 3 --max-total-programs-per-record 4000 --max-policy-candidates 128 --max-query-pool-cases 48
```

Observed on the 10-record smoke slice:

- Candidate oracle: `10/10`.
- Exact target synthesized: `10/10`.
- Visible-only hidden full-pass: `6/10`.
- Active max-split +1 query: `10/10`.
- Random +1 query: `15/20` over two repeats.

Interpretation: the full model-in-loop path is working, and the smoke directly exhibits the intended effect: active counterexample traces close a visible-selection ambiguity gap.

Next step: full ceiling eval with the same capped policy settings plus budget `12`.

## 2026-06-24 03:51 UTC

Full ceiling eval completed.

Command:

```bash
python scripts/eval_active_selection.py --data data/eval/dsl_eval_ceiling.jsonl --adapter /workspace/large_artifacts/qwen35_4b_active_counterexample_trace_selection/models/sketch_lora --sketch-source model --output reports/eval/active_ceiling.json --budgets 0,1,2,3,6,12 --random-repeats 2 --num-samples 3 --max-total-programs-per-record 4000 --max-policy-candidates 128 --max-query-pool-cases 48
```

Primary results:

- Candidate oracle: `120/120`.
- Exact target synthesized: `120/120`.
- Visible-only: `93/120` hidden full-pass.
- Active max-split +1: `103/120`.
- Active max-split +3: `110/120`.
- Active max-split +6: `116/120`.
- Active max-split +12: `117/120`.
- Oracle elimination +12: `118/120`.
- Random +12: `231/240` over two repeats.

Interpretation: active counterexample traces substantially close the selection gap on the primary split; the remaining gap is selection-policy quality, not candidate coverage.

Next step: support and IID retention evals with the same settings.

## 2026-06-24 04:27 UTC

Full support eval completed.

Command:

```bash
python scripts/eval_active_selection.py --data data/eval/dsl_eval_support.jsonl --adapter /workspace/large_artifacts/qwen35_4b_active_counterexample_trace_selection/models/sketch_lora --sketch-source model --output reports/eval/active_support.json --budgets 0,1,2,3,6,12 --random-repeats 2 --num-samples 3 --max-total-programs-per-record 4000 --max-policy-candidates 128 --max-query-pool-cases 48
```

Results:

- Candidate oracle: `120/120`.
- Exact target synthesized: `120/120`.
- Visible-only: `84/120`.
- Active max-split +1: `89/120`.
- Active max-split +3: `113/120`.
- Active max-split +6: `119/120`.
- Active max-split +12: `120/120`.
- Oracle elimination +6/+12: `120/120`.
- Random +12: `230/240` over two repeats.

Interpretation: active selection is not only a ceiling-split effect; it also closes nearly all support-selection ambiguity under the same policy.

Next step: IID retention eval.

## 2026-06-24 04:38 UTC

Full IID eval completed.

Command:

```bash
python scripts/eval_active_selection.py --data data/eval/dsl_eval_iid.jsonl --adapter /workspace/large_artifacts/qwen35_4b_active_counterexample_trace_selection/models/sketch_lora --sketch-source model --output reports/eval/active_iid.json --budgets 0,1,2,3,6,12 --random-repeats 2 --num-samples 3 --max-total-programs-per-record 4000 --max-policy-candidates 128 --max-query-pool-cases 48
```

Results:

- Candidate oracle: `60/60`.
- Exact target synthesized: `60/60`.
- Visible-only: `46/60`.
- Active max-split +1: `52/60`.
- Active max-split +3: `58/60`.
- Active max-split +6: `59/60`.
- Active max-split +12: `59/60`.
- Oracle elimination +2/+3/+6/+12: `59/60`.
- Random +12: `114/120` over two repeats.

Interpretation: the same active-query selection mechanism improves IID retention too; one record remains unresolved even under the oracle-elimination policy within the capped candidate/pool policy settings.

Next step: generate report, plots, and CSV summaries.

## 2026-06-24 04:40 UTC

Generated final reports and figures.

Command:

```bash
python scripts/make_report.py
```

Artifacts:

- Final report: `reports/qwen35_4b_active_counterexample_trace_selection_report.md`.
- Summary CSVs: `reports/policy_summary.csv`, `reports/candidate_summary.csv`.
- Figures:
  - `reports/figures/candidate_coverage.png`
  - `reports/figures/ceiling_success_by_budget.png`
  - `reports/figures/support_success_by_budget.png`
  - `reports/figures/iid_success_by_budget.png`
  - `reports/figures/split_comparison_budget6.png`
  - `reports/figures/split_comparison_budget12.png`
  - `reports/figures/ceiling_active_family_heatmap.png`

Manual post-generation edit: added key findings and method definitions to the report.
