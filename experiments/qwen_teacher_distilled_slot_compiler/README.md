# Qwen Teacher-Distilled Slot Compiler

**Status:** finished

Standalone experiment testing whether an oracle slot teacher improves a QLoRA-adapted `Qwen/Qwen3-4B` numeric-copy compiler on long modular-arithmetic programs.

The student compiler reads the full prompt hidden sequence, predicts ordered executable slots, copies numeric and operation symbols from token-level maps, and executes the copied program modulo 97. The teacher condition adds two auxiliary signals during training:

- soft local position targets around the oracle slot span;
- slot-representation matching between the student-attended slot vector and the oracle token vector.

## Main Result

Teacher slot distillation did not beat the matched no-teacher control on fresh length-24 paired programs. The control scored 27.1% exact execution and 72.7% paired compiler-state consistency. The heavier teacher arm scored 27.9% exact execution but only 55.5% consistency. The low-weight soft-position-only teacher arm scored 18.4% exact execution and 41.0% consistency.

The practical takeaway is that local oracle slot imitation is not the current bottleneck; the next experiment should target execution-level repair or verifier-guided search.

## Layout

```text
src/qwen_teacher_distilled_slot_compiler_experiment.py  training and evaluation harness
src/analyze_qwen_teacher_distilled_slot_compiler.py     aggregation, selected checkpoint table, and plots
src/evaluate_selected_qwen_teacher_distilled_slot_compiler.py
                                                        fresh retest for selected checkpoints
runs/                                                   lightweight JSON and CSV outputs
analysis/                                               aggregate CSVs and generated figures
reports/                                                experiment log and standalone write-up
checkpoint_manifest.csv                                 generated list of large checkpoint files
```

Large checkpoints are stored outside the experiment directory:

```text
large_artifacts/qwen_teacher_distilled_slot_compiler/checkpoints/
```

## Main Metrics

- `executor_accuracy`: exact final answer after copying and executing the compiled program.
- `program_exact`: exact full compiled-program correctness.
- `arg_accuracy`: per-argument copied value accuracy.
- `arg_pos_accuracy`: per-argument slot localization accuracy.
- `compiler_pair_state_consistency`: whether standard and paraphrased renderings compile to the same complete state trajectory.
- `selected_checkpoint`: saved checkpoint chosen by `paired_len24_executor_accuracy`.

## Reading Order

1. `reports/qwen_teacher_distilled_slot_compiler_experiment_log.md`
2. `analysis/summary.md`
3. `analysis/selected_checkpoints.csv`
4. `analysis/selected_retest_metrics.csv`
5. `analysis/final_metrics.csv`
6. `reports/qwen_teacher_distilled_slot_compiler_paper.md`
