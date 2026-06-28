# Experiment Log

## 2026-06-23

Created standalone on-policy repair-to-compiler experiment.

Initial design:

- Start from a QLoRA Qwen numeric compiler checkpoint.
- Generate targets from the current compiler policy, not from an independent editor.
- Enumerate local init/op/arg repairs around each emitted program.
- Use verified repairs when found; fall back to gold trace targets by default so the training set remains dense.
- Fine-tune the same compiler policy and evaluate fresh standard, paraphrase, and paired prompts after each round.
- Generate markdown and HTML reports with figures from the analyzer.

Initial implementation files:

- `src/qwen_onpolicy_repair_compiler_experiment.py`
- `src/qwen_onpolicy_repair_compiler_core.py`
- `analysis/analyze_qwen_onpolicy_repair_compiler.py`

Next steps:

- Compile the scripts.
- Copy the fixed compiler checkpoint into this experiment's large-artifact namespace.
- Run a small smoke test.
- If the smoke test passes, run a pilot with enough examples to detect whether on-policy targets improve the compiler or damage it.

### Smoke Run

Command shape:

```text
python experiments/qwen_onpolicy_repair_compiler/src/qwen_onpolicy_repair_compiler_experiment.py --run_name smoke_onpolicy_repair --train_examples 8 --val_examples 4 --eval_examples 4 --eval_pairs 4 --onpolicy_rounds 1 --epochs_per_round 1 --train_batch_size 1 --qwen_batch_size 2 --repair_topk 2 --repair_max_edits 1 --repair_max_pair_arg_slots 4 --max_length 384 --lr 2e-5 --gold_trace_loss_weight 0.05 --executor_loss_weight 0.0 --state_loss_weight 0.0 --seed 72 --eval_seed 72001
```

Outcome:

- Passed model loading, target generation, fine-tuning, evaluation, CSV/JSON writing, and checkpoint saving.
- Validation compiler accuracy stayed at 25.0% on the tiny smoke split.
- Validation local-repair ceiling was also 25.0% with the deliberately tiny repair budget.
- Fresh paired compiler accuracy was 0.0% on the tiny paired split, which is not meaningful beyond path validation.

### Pilot Run

Command shape:

```text
python experiments/qwen_onpolicy_repair_compiler/src/qwen_onpolicy_repair_compiler_experiment.py --run_name pilot_onpolicy_repair_s96_r1 --train_examples 96 --val_examples 64 --eval_examples 64 --eval_pairs 64 --onpolicy_rounds 1 --epochs_per_round 1 --train_batch_size 2 --qwen_batch_size 8 --repair_topk 3 --repair_max_edits 2 --repair_max_pair_arg_slots 8 --max_length 384 --lr 5e-5 --gold_trace_loss_weight 0.15 --executor_loss_weight 0.1 --state_loss_weight 0.0 --seed 73 --eval_seed 73001
```

Outcome:

- Baseline validation compiler accuracy was 32.8%; validation local-repair ceiling was 68.8%.
- After one on-policy round, validation compiler accuracy reached 90.6%.
- Fresh paired compiler accuracy reached 91.4%.
- Verified repair targets were found for 67.7% of training rows; 37.5% of targets changed at least one slot.

### Main Run

Command shape:

```text
python experiments/qwen_onpolicy_repair_compiler/src/qwen_onpolicy_repair_compiler_experiment.py --run_name main_onpolicy_repair_s256 --train_examples 256 --val_examples 128 --eval_examples 256 --eval_pairs 256 --onpolicy_rounds 1 --epochs_per_round 1 --train_batch_size 2 --qwen_batch_size 8 --repair_topk 3 --repair_max_edits 2 --repair_max_pair_arg_slots 8 --max_length 384 --lr 5e-5 --gold_trace_loss_weight 0.15 --executor_loss_weight 0.1 --state_loss_weight 0.0 --seed 74 --eval_seed 74001
```

Outcome:

- Baseline fresh paired compiler accuracy was 29.1%; baseline local-repair ceiling was 64.5%.
- After one round, fresh paired compiler accuracy reached 99.2%; local-repair ceiling reached 100.0%.
- Fresh standard reached 99.6%; fresh paraphrase reached 97.3%.
- Verified repair targets were found for 67.6% of training rows; 36.7% of targets changed at least one slot.

### Attribution Controls

Gold-only control:

```text
python experiments/qwen_onpolicy_repair_compiler/src/qwen_onpolicy_repair_compiler_experiment.py --run_name control_gold_only_s256 --train_examples 256 --val_examples 128 --eval_examples 256 --eval_pairs 256 --onpolicy_rounds 1 --epochs_per_round 1 --train_batch_size 2 --qwen_batch_size 8 --repair_topk 3 --repair_max_edits 2 --repair_max_pair_arg_slots 8 --max_length 384 --lr 5e-5 --target_mode gold_only --gold_trace_loss_weight 0.15 --executor_loss_weight 0.1 --state_loss_weight 0.0 --seed 74 --eval_seed 74001
```

Repair-only control:

```text
python experiments/qwen_onpolicy_repair_compiler/src/qwen_onpolicy_repair_compiler_experiment.py --run_name control_repair_only_s256 --train_examples 256 --val_examples 128 --eval_examples 256 --eval_pairs 256 --onpolicy_rounds 1 --epochs_per_round 1 --train_batch_size 2 --qwen_batch_size 8 --repair_topk 3 --repair_max_edits 2 --repair_max_pair_arg_slots 8 --max_length 384 --lr 5e-5 --target_mode repair_only --gold_trace_loss_weight 0.0 --executor_loss_weight 0.0 --state_loss_weight 0.0 --seed 74 --eval_seed 74001
```

Control outcome:

- Gold-only matched the mixed main run at 99.2% fresh paired accuracy.
- Repair-only reached 91.0% fresh paired accuracy with 173 active repaired rows and 83 skipped rows.
- Interpretation: the headline compiler improvement is real, but dense gold trace supervision is sufficient under this budget; verified local repairs alone are useful but weaker.

### Report Generation

Generated:

- `analysis/summary.md`
- `analysis/final_metrics.csv`
- `analysis/all_final_metrics.csv`
- `analysis/figures/*.png`
- `reports/qwen_onpolicy_repair_compiler_paper.md`
- `reports/qwen_onpolicy_repair_compiler_paper.html`
