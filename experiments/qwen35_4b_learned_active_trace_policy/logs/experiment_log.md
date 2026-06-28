# Experiment Log

## 2026-06-24 05:00 UTC

Initialized a standalone learned active trace policy experiment package.

Design commitments:

- Use `Qwen/Qwen3.5-4B`.
- Train a fresh sketch LoRA and a fresh policy LoRA for this experiment.
- Keep model adapters and checkpoints under `/workspace/large_artifacts/qwen35_4b_learned_active_trace_policy`.
- Keep downloadable experiment files under `/workspace/experiments/qwen35_4b_learned_active_trace_policy`.
- Train the policy by oracle-action distillation from candidate-program ambiguity states.
- Do not reveal query-option expected outputs to the policy prompt; only visible examples and candidate-output buckets are shown.
- Evaluate visible-only, random extra trace, active max-split, learned Qwen policy, and oracle elimination on IID, support, and ceiling splits.

Initial implementation:

- Added `src/active_core.py` for shared candidate-bank, query-option, prompt, policy, and summary logic.
- Added `scripts/build_policy_dataset.py` for oracle-distilled policy SFT examples.
- Added `scripts/train_policy_adapter.py` for the Qwen policy LoRA.
- Added `scripts/eval_learned_policy.py` for model-in-loop learned-policy evaluation.
- Replaced `scripts/make_report.py` with a learned-policy report generator.

Validation:

- `python -m py_compile scripts/build_dataset.py scripts/build_policy_dataset.py scripts/train_adapter.py scripts/train_policy_adapter.py scripts/eval_learned_policy.py scripts/make_report.py src/*.py`
- Result: passed.
- Removed generated `__pycache__` directories from the experiment tree.

## 2026-06-24 05:05 UTC

Built the standalone DSL dataset.

Command:

```bash
python scripts/build_dataset.py > run_logs/dataset_build_console.log 2>&1
```

Observed:

- Static80 sketch-training records: `240`.
- Seed/policy-validation records: `240`.
- IID eval records: `60`.
- Support eval records: `120`.
- Ceiling eval records: `120`.
- Active query pool before policy cap: `384` cases per eval record.

Next check: build a small policy-distillation smoke dataset, inspect labels and prompt shape, then build the full policy SFT dataset.

## 2026-06-24 05:11 UTC

Policy-distillation smoke iteration 1 produced zero examples.

Command:

```bash
python scripts/build_policy_dataset.py --train-data data/static_bridge_80/dsl_train.jsonl --eval-data data/seed/dsl_train.jsonl --out-dir data/policy_smoke --max-train-records 12 --max-eval-records 8 --max-steps 3 --max-options 24 --max-policy-candidates 128 --max-query-pool-cases 48 --max-total-programs-per-record 4000 > run_logs/policy_dataset_smoke_console.log 2>&1
```

Cause:

- The first dataset build attached `case_pool` only to eval records.
- Policy SFT needs query pools on training records too.

Fix:

- Patched `src/data_gen.py` so `static_bridge_records` accepts `case_pool_count`.
- Patched `scripts/build_dataset.py` to attach active case pools to seed, static-base, and bridge training records.
- Rebuilt the dataset with `python scripts/build_dataset.py > run_logs/dataset_build_v2_console.log 2>&1`.

Verification:

- `data/static_bridge_80/dsl_train.jsonl` first record has `384` query-pool cases.
- `data/seed/dsl_train.jsonl` first record has `384` query-pool cases.
- `data/dataset_manifest.json` records `train_records_include_active_case_pools: true`.

## 2026-06-24 05:23 UTC

Policy-distillation smoke iteration 2 succeeded.

Command:

```bash
python scripts/build_policy_dataset.py --train-data data/static_bridge_80/dsl_train.jsonl --eval-data data/seed/dsl_train.jsonl --out-dir data/policy_smoke --max-train-records 12 --max-eval-records 8 --max-steps 3 --max-options 24 --max-policy-candidates 128 --max-query-pool-cases 48 --max-total-programs-per-record 4000 > run_logs/policy_dataset_smoke_v2_console.log 2>&1
```

Observed:

- Train examples: `8`.
- Eval examples: `2`.
- First prompt displayed 24 options and hid the query-option expected output.
- First label: `Q00`, with `2` actual candidate eliminations.

Decision:

- Use both static bridge training files for the full policy source to increase supervised states.
- Keep seed records as policy-validation source.

## 2026-06-24 05:34 UTC

Built the full oracle-distilled policy dataset.

Command:

```bash
python scripts/build_policy_dataset.py --train-data data/static_bridge_80/dsl_train.jsonl data/static_bridge_60/dsl_train.jsonl --eval-data data/seed/dsl_train.jsonl --out-dir data/policy --max-eval-records 120 --max-steps 3 --max-options 24 --max-policy-candidates 128 --max-query-pool-cases 48 --max-total-programs-per-record 4000 > run_logs/policy_dataset_full_console.log 2>&1
```

Observed:

- Train source records: `480`.
- Policy train examples: `481`.
- Policy eval examples: `95`.
- Train examples by step: `241` at step 1, `165` at step 2, `75` at step 3.
- Average target actual eliminations: `27.744`.
- Target action distribution is nontrivial: `Q00` occurs `204/481`, with many labels deeper in the displayed list.
- Token length check with the Qwen tokenizer: max `2882`, p95 `2673`, `0` examples above `4096`.

Next step: train a fresh sketch LoRA, then train the policy LoRA.

## 2026-06-24 05:54 UTC

Sketch LoRA training completed successfully.

Command:

```bash
python scripts/train_adapter.py --train data/static_bridge_80/dsl_train.jsonl --eval data/eval/dsl_eval_iid.jsonl --task sketch --target-field target_sketch --prompt-mode trace --output-dir /workspace/large_artifacts/qwen35_4b_learned_active_trace_policy/models/sketch_lora --epochs 2.0 --lr 1.5e-4 --rank 32 --alpha 64 --dropout 0.05 --grad-accum 8 --save-steps 30 --eval-steps 30 > run_logs/training_sketch_lora_console.log 2>&1
```

Observed:

- Trainable parameters: `42,467,328`.
- Step 30 eval loss: `0.0001971`.
- Final eval loss at epoch 2: `0.000829`.
- Adapter/checkpoint tree size: `709M`.
- Large artifact location: `/workspace/large_artifacts/qwen35_4b_learned_active_trace_policy/models/sketch_lora`.

Next step: train the Qwen active-query policy LoRA on `data/policy/policy_train.jsonl`.

## 2026-06-24 06:37 UTC

Policy LoRA training was run with checkpointed validation and stopped early after the validation loss stopped improving.

Command:

```bash
python scripts/train_policy_adapter.py --train data/policy/policy_train.jsonl --eval data/policy/policy_eval.jsonl --output-dir /workspace/large_artifacts/qwen35_4b_learned_active_trace_policy/models/policy_lora --epochs 3.0 --lr 1.2e-4 --rank 32 --alpha 64 --dropout 0.05 --grad-accum 8 --save-steps 40 --eval-steps 40 --max-eval-records 95 > run_logs/training_policy_lora_console.log 2>&1
```

Observed:

- Trainable parameters: `42,467,328`.
- Checkpoint 40 eval loss: `0.399913`.
- Checkpoint 80 eval loss: `0.405868`.
- Checkpoint 80 was worse than checkpoint 40, so training was intentionally interrupted at step 81 rather than spending another long epoch on likely overfit/noise.
- Selected policy checkpoint for evaluation: `/workspace/large_artifacts/qwen35_4b_learned_active_trace_policy/models/policy_lora/checkpoint-40`.
- Policy artifact tree size after two checkpoints: `528M`.

Next step: run target-sketch smoke evaluation with the selected policy checkpoint to verify parsing/action behavior, then run full model-sketch evaluations.

## 2026-06-24 06:40 UTC

Target-sketch learned-policy smoke evaluation completed.

Command:

```bash
python scripts/eval_learned_policy.py --data data/eval/dsl_eval_ceiling.jsonl --sketch-source target --policy-adapter /workspace/large_artifacts/qwen35_4b_learned_active_trace_policy/models/policy_lora/checkpoint-40 --output reports/eval/_smoke_target_policy_ceiling10.json --max-records 10 --budgets 0,1,2,3 --random-repeats 1 --max-total-programs-per-record 4000 --max-policy-candidates 128 --max-query-pool-cases 48 --policy-max-options 24 > run_logs/eval_smoke_target_policy_ceiling10_console.log 2>&1
```

Observed:

- Candidate oracle: `10/10`.
- Exact target synthesized: `10/10`.
- Visible-only: `10/10`.
- Learned policy parse rate: `1.0`.
- Learned traces: `60` parseable actions, `0` fallbacks.
- The smoke slice was already solved, so it validates mechanics and parsing but not policy advantage.

Next step: run a model-sketch smoke where visible selection is expected to be less saturated.

## 2026-06-24 06:45 UTC

Full-path model-sketch learned-policy smoke evaluation completed.

Command:

```bash
python scripts/eval_learned_policy.py --data data/eval/dsl_eval_ceiling.jsonl --sketch-source model --sketch-adapter /workspace/large_artifacts/qwen35_4b_learned_active_trace_policy/models/sketch_lora --policy-adapter /workspace/large_artifacts/qwen35_4b_learned_active_trace_policy/models/policy_lora/checkpoint-40 --output reports/eval/_smoke_model_policy_ceiling10.json --max-records 10 --budgets 0,1,2,3 --random-repeats 1 --num-samples 3 --max-total-programs-per-record 4000 --max-policy-candidates 128 --max-query-pool-cases 48 --policy-max-options 24 > run_logs/eval_smoke_model_policy_ceiling10_console.log 2>&1
```

Observed:

- Candidate oracle: `10/10`.
- Exact target synthesized: `10/10`.
- Visible-only: `10/10`.
- Learned policy parse rate: `1.0`.
- Learned traces: `60` parseable actions, `0` fallbacks.
- Both Qwen adapters loaded and ran together successfully.

Decision:

- Full eval will use budgets `0,1,2,3`.
- Reason: the policy SFT data contains oracle trajectories through step 3, and the main experimental question is low-budget learned control.

## 2026-06-24 07:18 UTC

Full ceiling evaluation completed.

Command:

```bash
python scripts/eval_learned_policy.py --data data/eval/dsl_eval_ceiling.jsonl --sketch-source model --sketch-adapter /workspace/large_artifacts/qwen35_4b_learned_active_trace_policy/models/sketch_lora --policy-adapter /workspace/large_artifacts/qwen35_4b_learned_active_trace_policy/models/policy_lora/checkpoint-40 --output reports/eval/learned_ceiling.json --budgets 0,1,2,3 --random-repeats 2 --num-samples 3 --max-total-programs-per-record 4000 --max-policy-candidates 128 --max-query-pool-cases 48 --policy-max-options 24 > run_logs/eval_learned_ceiling_console.log 2>&1
```

Observed:

- Candidate oracle: `120/120`.
- Exact target synthesized: `120/120`.
- Visible-only: `101/120`.
- Active max-split: `109/120` at +1, `113/120` at +2, `115/120` at +3.
- Learned Qwen policy: `104/120` at +1, `112/120` at +2, `115/120` at +3.
- Oracle elimination: `116/120` at +1, `120/120` at +2 and +3.
- Learned parse rate: `1.0`.

Interpretation before retention checks:

- The learned policy is mechanically reliable and catches up to active max-split by +3.
- It does not beat active max-split in the highest-leverage +1 regime on the primary split.
- The oracle gap remains large at +1, so the policy-learning formulation is still relevant, but the current SFT controller has not solved it.

Next step: run support and IID retention evaluations with the same settings.

## 2026-06-24 07:50 UTC

Full support evaluation completed.

Command:

```bash
python scripts/eval_learned_policy.py --data data/eval/dsl_eval_support.jsonl --sketch-source model --sketch-adapter /workspace/large_artifacts/qwen35_4b_learned_active_trace_policy/models/sketch_lora --policy-adapter /workspace/large_artifacts/qwen35_4b_learned_active_trace_policy/models/policy_lora/checkpoint-40 --output reports/eval/learned_support.json --budgets 0,1,2,3 --random-repeats 2 --num-samples 3 --max-total-programs-per-record 4000 --max-policy-candidates 128 --max-query-pool-cases 48 --policy-max-options 24 > run_logs/eval_learned_support_console.log 2>&1
```

Observed:

- Candidate oracle: `120/120`.
- Exact target synthesized: `120/120`.
- Visible-only: `72/120`.
- Active max-split: `83/120` at +1, `93/120` at +2, `107/120` at +3.
- Learned Qwen policy: `85/120` at +1, `99/120` at +2, `104/120` at +3.
- Oracle elimination: `103/120` at +1, `115/120` at +2, `116/120` at +3.
- Learned parse rate: `1.0`.

Interpretation:

- Learned policy beats active max-split at +1 and +2 on support retention.
- Learned policy falls behind active max-split by +3, suggesting the learned policy can choose useful early discriminators but compounds less reliably across multiple queries.

Next step: run IID retention evaluation.

## 2026-06-24 08:02 UTC

Full IID evaluation completed.

Command:

```bash
python scripts/eval_learned_policy.py --data data/eval/dsl_eval_iid.jsonl --sketch-source model --sketch-adapter /workspace/large_artifacts/qwen35_4b_learned_active_trace_policy/models/sketch_lora --policy-adapter /workspace/large_artifacts/qwen35_4b_learned_active_trace_policy/models/policy_lora/checkpoint-40 --output reports/eval/learned_iid.json --budgets 0,1,2,3 --random-repeats 2 --num-samples 3 --max-total-programs-per-record 4000 --max-policy-candidates 128 --max-query-pool-cases 48 --policy-max-options 24 > run_logs/eval_learned_iid_console.log 2>&1
```

Observed:

- Candidate oracle: `60/60`.
- Exact target synthesized: `60/60`.
- Visible-only: `47/60`.
- Active max-split: `51/60` at +1, `57/60` at +2, `58/60` at +3.
- Learned Qwen policy: `53/60` at +1, `55/60` at +2, `55/60` at +3.
- Oracle elimination: `55/60` at +1, `56/60` at +2, `58/60` at +3.
- Learned parse rate: `1.0`.

Interpretation:

- Learned policy improves the single-query regime on IID.
- It compounds worse than active max-split after additional queries.
- Across splits, this points to a real learned first-query signal but insufficient recurrent/state-update robustness.

Next step: generate report, plots, and CSV summaries.

## 2026-06-24 08:04 UTC

Generated report, figures, and CSV summaries.

Command:

```bash
python scripts/make_report.py > run_logs/report_generation_console.log 2>&1
```

Artifacts:

- Final report: `reports/qwen35_4b_learned_active_trace_policy_report.md`.
- Summary CSVs: `reports/policy_summary.csv`, `reports/candidate_summary.csv`, `reports/learned_parse_summary.csv`, `reports/learned_query_trace_summary.csv`.
- Figures:
  - `reports/figures/candidate_coverage.png`
  - `reports/figures/ceiling_success_by_budget.png`
  - `reports/figures/support_success_by_budget.png`
  - `reports/figures/iid_success_by_budget.png`
  - `reports/figures/split_comparison_budget1.png`
  - `reports/figures/split_comparison_budget3.png`
  - `reports/figures/oracle_gap_budget1.png`
  - `reports/figures/oracle_gap_budget3.png`
  - `reports/figures/learned_choice_rank.png`
  - `reports/figures/policy_dataset_examples_by_step.png`

Manual report edit after generation:

- Added key findings and training notes to make the writeup reflect the actual result: reliable parsing, real +1 gains on support/IID, no +1 gain on ceiling, and weak multi-step compounding.

## 2026-06-24 08:10 UTC

Final packaging and audit completed.

Checks:

- Removed generated `__pycache__` directories.
- Removed an unused copied evaluator script that belonged to a different experiment package.
- Searched the final experiment tree for stale old-experiment references; no matches remained.
- Verified that no file larger than `50M` remains under `/workspace/experiments/qwen35_4b_learned_active_trace_policy`.
- Verified experiment package size: `85M`.
- Verified large artifact package size: `1.3G`.
- Verified all 10 generated PNG figures are readable and non-empty.

Final locations:

- Downloadable experiment package: `/workspace/experiments/qwen35_4b_learned_active_trace_policy`.
- Large adapters/checkpoints: `/workspace/large_artifacts/qwen35_4b_learned_active_trace_policy`.
- Final report: `reports/qwen35_4b_learned_active_trace_policy_report.md`.
