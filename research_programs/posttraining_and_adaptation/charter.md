# Posttraining And Adaptation

## Purpose

Study how small-model behavior changes under lightweight updates: LoRA, QLoRA, DPO, distillation, DAgger, GRPO, preference objectives, and process supervision.

## Why This Is A Program

The repo should support many adaptation mechanisms, not treat any one method as default. The central question is which update target changes the model in the desired direction without destroying generality or safety.

## Progress Signals

- Updates beat frozen baselines and non-mechanistic controls.
- Gains transfer beyond training-like tasks.
- Training artifacts are reproducible without checking adapter weights into git.
- The update objective explains the behavioral change.

## Boundaries

This program owns model updates. Candidate generation, selection, or tool control can be downstream beneficiaries.
