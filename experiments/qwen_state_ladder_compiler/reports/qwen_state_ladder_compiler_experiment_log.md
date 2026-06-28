# Qwen State-Ladder Compiler Experiment Log

## Objective

Test whether dense supervision of the latent modular state after every operation improves a Qwen numeric-copy compiler's long-chain exact execution.

## Experiment Question

Can a QLoRA-adapted model learn a more reliable executable compiler interface when training includes a curriculum over chain lengths and a per-step state target, rather than only final-answer executor loss plus symbol trace supervision?

## Planned Conditions

1. Smoke tests for intermediate-state targets, differentiable state trajectories, and curriculum stage handling.
2. Main QLoRA numeric-copy compiler with trace loss, executor loss, and state-ladder loss.
3. Matched QLoRA numeric-copy compiler with the same curriculum but without state-ladder loss.
4. Final-answer-only QLoRA control with the same model and total step budget.

## Primary Metrics

- `executor_accuracy`: exact final answer after copying and running the compiled program.
- `program_exact`: exact full compiled-program correctness.
- `state_accuracy`: per-step intermediate-state accuracy.
- `state_all_exact`: whether every intermediate state in a program is correct.
- `state_prefix_fraction`: fraction of steps before the first state error, aggregated across examples.
- `compiler_pair_state_consistency`: whether paired renderings compile to the same state trajectory.
- `init_accuracy`, `op_accuracy`, `arg_accuracy`: copied symbol correctness.
- `op_pos_accuracy`, `arg_pos_accuracy`: ordered slot localization.

## Artifact Policy

Lightweight outputs stay in `experiments/qwen_state_ladder_compiler/runs/`.

Large adapters and head checkpoints stay in `large_artifacts/qwen_state_ladder_compiler/checkpoints/`.

## Log

### 2026-06-21

- Created standalone experiment directory and external checkpoint path.
- Forked the numeric-copy compiler harness into a state-ladder harness.
- Added gold intermediate state targets, differentiable state trajectories, state-ladder loss, state trajectory diagnostics, and curriculum stage parsing.
- Ran tiny-model smoke tests for the state-ladder and no-state variants. Both completed and produced state metrics.
- Ran a Qwen3-4B QLoRA smoke test for the state-ladder path. The 4-bit LoRA path, trajectory loss, and paired evaluation all worked.
- Ran a 400-step Qwen pilot with full state-ladder loss and a matched no-state-ladder control. The pilot showed that the curriculum itself was powerful and that state loss improved paired/paraphrase trajectory metrics at short budget.
- Ran full 900-step Qwen conditions:
  - `main_qwen3_4b_qlora_state_ladder_curriculum_s900`
  - `control_qwen3_4b_qlora_curriculum_no_state_ladder_s900`
  - `control_qwen3_4b_qlora_answer_only_curriculum_s900`
- The full-weight state ladder improved standard L12 but hurt final L24, especially paraphrase and paired L24.
- Ran a targeted lighter-loss ablation:
  - `main_qwen3_4b_qlora_state_ladder_w025_curriculum_s900`
- The lighter state ladder reduced the damage and had the best logged paired L24 checkpoint at step 800, but its final checkpoint still trailed the no-state curriculum control on paired L24.
- Final result: the staged length curriculum is the main positive finding. Dense state-ladder loss is not a robust fixed-weight improvement under these settings.
