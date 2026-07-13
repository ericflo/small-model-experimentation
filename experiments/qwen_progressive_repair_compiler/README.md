# Qwen Progressive Repair Compiler

**Status:** finished

This experiment trains a non-oracle repair selector for a frozen Qwen-attached
numeric program compiler. Each prompt is compiled into an executable
modular-arithmetic program, local candidate repairs are enumerated, and a small
transformer verifier chooses among candidate execution traces without access to
the true answer or true state trajectory at test time.

The experiment-specific small files live here:

```text
experiments/qwen_progressive_repair_compiler/
```

Large checkpoints live separately:

```text
large_artifacts/qwen_progressive_repair_compiler/checkpoints/
```

## Primary Command

```bash
python experiments/qwen_progressive_repair_compiler/src/qwen_progressive_repair_compiler_experiment.py \
  --run_name main_progressive_repair_s512 \
  --train_examples 512 \
  --val_examples 128 \
  --eval_examples 256 \
  --eval_pairs 256 \
  --candidate_curriculum small:2:1:8:3,medium:3:1:16:4,full:3:2:24:11
```

## Analysis

```bash
python experiments/qwen_progressive_repair_compiler/src/analyze_qwen_progressive_repair_compiler.py
```

Expected outputs:

- `analysis/summary.md`
- `analysis/final_metrics.csv`
- `analysis/all_final_metrics.csv`
- `analysis/figures/executor_accuracy.png`
- `analysis/figures/paired_details.png`
- `analysis/figures/training_curve.png`
- `reports/qwen_progressive_repair_compiler_paper.md`
- `reports/qwen_progressive_repair_compiler_paper.html`
- `checkpoint_manifest.csv`
