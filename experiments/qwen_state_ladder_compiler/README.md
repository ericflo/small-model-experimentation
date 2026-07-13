# Qwen State-Ladder Compiler

**Status:** finished

Standalone intermediate-state ladder experiment for a Qwen numeric-copy compiler.

The experiment trains a numeric-copy compiler on modular-arithmetic programs and adds dense supervision for the latent modular state after each operation. The main comparison tests whether that per-step state ladder improves long-chain exact execution over a matched trace-supervised compiler without the state loss.

The completed Qwen3-4B runs show that the staged length curriculum is strongly useful, while fixed-weight state-ladder loss is mixed. A lighter state loss gives the best logged paired length-24 checkpoint, but the final saved no-state curriculum control is strongest on paired length-24 exact execution.

## Layout

```text
src/qwen_state_ladder_compiler_experiment.py     training and evaluation harness
src/analyze_qwen_state_ladder_compiler.py        run aggregation and plots
runs/                                           lightweight JSON and CSV outputs
reports/                                        experiment log and standalone write-up
analysis/                                       aggregate tables and figures
```

Large checkpoints are stored outside the experiment directory:

```text
large_artifacts/qwen_state_ladder_compiler/checkpoints/
```

## Reading Order

1. `reports/qwen_state_ladder_compiler_paper.md`
2. `analysis/summary.md`
3. `analysis/final_metrics.csv`
4. `reports/qwen_state_ladder_compiler_experiment_log.md`

## Main Runs

```text
runs/main_qwen3_4b_qlora_state_ladder_curriculum_s900/
runs/main_qwen3_4b_qlora_state_ladder_w025_curriculum_s900/
runs/control_qwen3_4b_qlora_curriculum_no_state_ladder_s900/
runs/control_qwen3_4b_qlora_answer_only_curriculum_s900/
```
