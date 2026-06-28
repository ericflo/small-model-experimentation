# Qwen State-Ladder Numeric-Copy Compiler

## Abstract

This experiment tests whether dense intermediate-state supervision improves a QLoRA-adapted `Qwen/Qwen3-4B` numeric-copy compiler on long modular-arithmetic programs. Each prompt describes an initial value and a sequence of add, subtract, and multiply updates modulo 97. A trainable compiler reads Qwen hidden states, selects token positions for the initial value, operations, and arguments, copies exact symbols from deterministic token maps, and executes the copied program with an invisible modular runtime.

The main intervention is a state ladder: in addition to trace supervision and final-answer executor loss, the compiler is trained to match the latent modular state after each operation. Training uses a four-stage length curriculum: 1-4 steps, 1-8 steps, 1-12 steps, and 8-24 steps.

The main result is mixed. The curriculum itself is strongly useful: the matched no-state-ladder compiler reaches 39.1% exact execution on standard length 24, 15.6% on paraphrase length 24, and 21.1% on paired length 24. Final-answer-only QLoRA remains at chance. The state-ladder loss does not clearly beat the matched curriculum control at final checkpoint. A full-weight state ladder improves some shorter and medium splits but hurts length-24 robustness. A lighter state loss nearly matches the control and has the best logged paired length-24 checkpoint, but its final checkpoint still trails the control on paired length 24.

## Setup

- Base model: `Qwen/Qwen3-4B`
- Loader: `AutoModelForCausalLM`
- Quantization: 4-bit NF4
- Trainable update: LoRA rank 8, alpha 16, dropout 0.05, target `all-linear`
- Trainable LoRA parameters: 16,515,072
- Compiler head width: 768
- Task: modular arithmetic programs modulo 97
- Curriculum: `short:1:4:200`, `medium:1:8:200`, `train:1:12:200`, `long:8:24:300`
- Eval lengths: 4, 8, 12, and 24 update steps
- Eval templates: standard, paraphrase, and paired standard/paraphrase renderings of the same program
- Eval size: 64 examples per unpaired split; 64 program pairs per paired split
- Hardware: NVIDIA RTX 6000 Ada Generation, 48 GB class VRAM
- Large checkpoints: `large_artifacts/qwen_state_ladder_compiler/checkpoints/`

The compiler predicts:

- the token position of the initial value;
- ordered operation token positions;
- ordered argument token positions.

Values and operations are copied from per-token maps. The copied program is executed exactly modulo 97. For state-ladder training, the differentiable executor also returns a distribution over the modular state after every active step, and the training loss includes per-step NLL against the true state trajectory.

## Conditions

| Run | Variant | Purpose |
|---|---|---|
| `main_qwen3_4b_qlora_state_ladder_curriculum_s900` | `copy_trace_state_ladder` | Full state-ladder loss weight 1.0. |
| `main_qwen3_4b_qlora_state_ladder_w025_curriculum_s900` | `copy_trace_state_ladder` | Lighter state-ladder loss weight 0.25. |
| `control_qwen3_4b_qlora_curriculum_no_state_ladder_s900` | `copy_trace` | Matched curriculum compiler without state-ladder loss. |
| `control_qwen3_4b_qlora_answer_only_curriculum_s900` | `direct` | Final-answer-only QLoRA control without compiler structure. |

All compiler arms use the same explicit training seed (`2427`), same curriculum, same paired training distribution, same LoRA configuration, and same evaluation splits.

## Final Exact Execution

| Split | State Ladder 1.0 | State Ladder 0.25 | No State Ladder | Answer-Only QLoRA |
|---|---:|---:|---:|---:|
| Standard L4 | 93.8% | 90.6% | 92.2% | 0.0% |
| Standard L8 | 67.2% | 64.1% | 65.6% | 0.0% |
| Standard L12 | 51.6% | 45.3% | 43.8% | 1.6% |
| Standard L24 | 29.7% | 37.5% | 39.1% | 3.1% |
| Paraphrase L4 | 84.4% | 75.0% | 78.1% | 0.0% |
| Paraphrase L8 | 64.1% | 59.4% | 64.1% | 0.0% |
| Paraphrase L12 | 56.2% | 57.8% | 59.4% | 0.0% |
| Paraphrase L24 | 0.0% | 14.1% | 15.6% | 1.6% |
| Paired L4 | 86.7% | 86.7% | 86.7% | 1.6% |
| Paired L8 | 62.5% | 60.9% | 58.6% | 1.6% |
| Paired L12 | 53.1% | 59.4% | 59.4% | 0.0% |
| Paired L24 | 14.8% | 19.5% | 21.1% | 1.6% |

The answer-only control stays near 97-way chance. The compiler arms therefore owe their gains to the structured compiler and curriculum, not to ordinary final-answer fitting.

The no-state-ladder curriculum control is the strongest final checkpoint on the hardest target, paired length 24. Full-weight state loss helps standard length 12 but damages long paraphrase and paired performance. The lighter state loss is less damaging and reaches 37.5% standard L24, but still trails the no-state control on final paired L24.

## State Trajectory Metrics

`state_all_exact` measures whether every intermediate modular state is correct. `state_prefix_fraction` measures how far execution gets before the first state error, as a fraction of all active steps.

| Split | State 1.0 All Exact | State 0.25 All Exact | Control All Exact | State 1.0 Prefix | State 0.25 Prefix | Control Prefix |
|---|---:|---:|---:|---:|---:|---:|
| Standard L12 | 51.6% | 45.3% | 43.8% | 76.2% | 71.0% | 70.6% |
| Standard L24 | 28.1% | 37.5% | 39.1% | 62.8% | 63.7% | 63.7% |
| Paraphrase L12 | 56.2% | 57.8% | 59.4% | 74.7% | 72.4% | 74.2% |
| Paraphrase L24 | 0.0% | 14.1% | 15.6% | 40.0% | 46.1% | 45.2% |
| Paired L12 | 53.1% | 57.8% | 57.8% | 78.1% | 78.5% | 78.0% |
| Paired L24 | 14.8% | 18.8% | 20.3% | 49.8% | 49.0% | 48.6% |

The state metrics agree with the exact-execution result. The state ladder does not produce a clear final-checkpoint improvement on long programs. Weight 1.0 gives the best standard L12 trajectory but is too brittle at paraphrase L24. Weight 0.25 is more stable but still does not beat the no-state control at final L24.

## Paired State Consistency

Paired splits render the same latent program twice, once in the standard template and once in a paraphrased template. `compiler_pair_state_consistency` measures whether the two renderings produce the same complete predicted state trajectory.

| Split | State Ladder 1.0 | State Ladder 0.25 | No State Ladder |
|---|---:|---:|---:|
| Paired L4 | 98.4% | 95.3% | 98.4% |
| Paired L8 | 84.4% | 87.5% | 84.4% |
| Paired L12 | 71.9% | 76.6% | 73.4% |
| Paired L24 | 1.6% | 26.6% | 42.2% |

The full-weight state ladder collapses on paired L24 consistency at the final checkpoint. The lighter state ladder is better, but the matched no-state curriculum control still has the highest final paired L24 consistency.

## Best Logged Checkpoints

The final checkpoint is the primary saved model for each run. Training logs also show that paired L24 peaked before the final checkpoint.

| Run | Step | Paired L24 Exec | Paired L24 Prefix | Paired L24 State Consistency | Standard L24 Exec | Paraphrase L24 Exec |
|---|---:|---:|---:|---:|---:|---:|
| No state ladder | 800 | 25.0% | 51.1% | 53.1% | 32.8% | 23.4% |
| State ladder 1.0 | 800 | 23.4% | 52.3% | 43.8% | 31.2% | 10.9% |
| State ladder 0.25 | 800 | 28.1% | 52.7% | 56.2% | 35.9% | 21.9% |

The lighter state ladder has the best logged paired L24 checkpoint. However, those intermediate weights were not saved in this run; this table is a training-log result, not a saved-checkpoint result. The final saved checkpoint comparison remains mixed and favors the no-state control on paired L24.

## Interpretation

The positive result is the length curriculum. It pushes the numeric-copy compiler into a much stronger long-chain regime while final-answer-only QLoRA remains at chance. The no-state curriculum compiler reaches 39.1% standard L24 and 21.1% paired L24 exact execution with only 900 optimizer steps.

The state-ladder hypothesis is not cleanly supported. Dense intermediate-state loss changes learning dynamics and can improve some medium splits, but at the tested weights it does not reliably improve final long-chain execution over the matched curriculum control. The weight-0.25 ablation is informative: lowering the state loss reduces the damage and gives the best logged paired L24 checkpoint, but still does not produce a final-checkpoint win.

The strongest conclusion is therefore:

1. A staged long-chain curriculum is a high-leverage way to improve the Qwen numeric-copy compiler.
2. Dense state supervision is useful diagnostically and may help with checkpoint timing, but the tested fixed-weight objective is not yet the right long-chain regularizer.
3. The remaining failure is still compounding exactness: per-step symbol metrics are high, but length-24 exact execution remains sensitive to small operation and argument errors.

## Recommended Follow-Up

The next run should make checkpoint selection and state loss scheduling explicit:

- save checkpoints at each stage boundary and every long-stage evaluation;
- select by paired length-24 validation exactness, not final step;
- anneal state loss from 1.0 to 0.0 during the long stage, or use state loss only through length 12 and disable it for length 24;
- run a matched no-state curriculum control with the same checkpoint-selection rule.

The success criterion should be a saved checkpoint that beats the no-state curriculum control on paired L24 exact execution and paired L24 state consistency.

## Artifacts

Small files:

- Source: `experiments/qwen_state_ladder_compiler/src/`
- Runs: `experiments/qwen_state_ladder_compiler/runs/`
- Analysis: `experiments/qwen_state_ladder_compiler/analysis/`
- Report: `experiments/qwen_state_ladder_compiler/reports/qwen_state_ladder_compiler_paper.md`
- Manifest: `experiments/qwen_state_ladder_compiler/checkpoint_manifest.csv`

Large files:

- State ladder weight 1.0: `large_artifacts/qwen_state_ladder_compiler/checkpoints/main_qwen3_4b_qlora_state_ladder_curriculum_s900/`
- State ladder weight 0.25: `large_artifacts/qwen_state_ladder_compiler/checkpoints/main_qwen3_4b_qlora_state_ladder_w025_curriculum_s900/`
- No-state curriculum control: `large_artifacts/qwen_state_ladder_compiler/checkpoints/control_qwen3_4b_qlora_curriculum_no_state_ladder_s900/`
- Answer-only control: `large_artifacts/qwen_state_ladder_compiler/checkpoints/control_qwen3_4b_qlora_answer_only_curriculum_s900/`
