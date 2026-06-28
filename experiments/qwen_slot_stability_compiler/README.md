# Qwen Slot-Stability Compiler

Standalone paired paraphrase slot-stability experiment for a Qwen numeric-copy compiler.

The experiment trains a numeric-copy compiler on paired renderings of the same modular-arithmetic program. The stability condition penalizes drift in copied slot distributions across paired wording variants while preserving trace and executor supervision.

The completed 600-step Qwen3-4B runs show that trace-supervised numeric-copy compilation works far above chance, while answer-only QLoRA stays at chance. The explicit paired stability loss is a mixed regularizer rather than a clean win over the matched paired-data trace compiler.

## Layout

```text
src/qwen_slot_stability_compiler_experiment.py     training and evaluation harness
src/analyze_qwen_slot_stability_compiler.py        run aggregation and plots
runs/                                             lightweight JSON and CSV outputs
reports/                                          experiment log and standalone write-up
analysis/                                         aggregate tables and figures
```

Large checkpoints are stored outside the experiment directory:

```text
large_artifacts/qwen_slot_stability_compiler/checkpoints/
```

## Reading Order

1. `reports/qwen_slot_stability_compiler_paper.md`
2. `analysis/summary.md`
3. `analysis/final_metrics.csv`
4. `reports/qwen_slot_stability_compiler_experiment_log.md`

## Main Runs

```text
runs/main_qwen3_4b_qlora_slot_stability_mixed_l12_s600/
runs/control_qwen3_4b_qlora_paired_no_stability_mixed_l12_s600/
runs/control_qwen3_4b_qlora_answer_only_mixed_l12_s600/
```
