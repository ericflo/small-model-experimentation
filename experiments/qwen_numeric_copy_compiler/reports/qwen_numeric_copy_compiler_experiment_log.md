# Qwen Numeric-Copy Compiler Experiment Log

## Objective

Test whether a Qwen model can expose a symbolic modular-arithmetic program through learned token roles while exact numeric and operator values are copied from lexical token maps instead of inferred by semantic value classifiers.

## Experiment Question

Can a frozen hidden-state parser pass a numeric-copy pilot gate, and does live QLoRA training improve or preserve that compiler interface on standard and paraphrased arithmetic programs?

## Planned Gates

1. Frozen numeric-copy pilot: train compiler heads over frozen model hidden states, with the backbone kept fixed.
2. QLoRA numeric-copy compiler: train LoRA adapters and compiler heads live.
3. Controls: final-answer direct adaptation and answer-only compiler training.

## Primary Metrics

- `init_pos_accuracy`, `op_pos_accuracy`, `arg_pos_accuracy`: role and slot localization.
- `init_accuracy`, `op_accuracy`, `arg_accuracy`: copied program-symbol correctness.
- `program_exact`: exact compiled program correctness.
- `executor_accuracy`: exact modular execution accuracy.
- Standard and paraphrase splits are evaluated separately at multiple chain lengths.

## Artifact Policy

Lightweight outputs stay in `experiments/qwen_numeric_copy_compiler/runs/`.

Large adapters and head checkpoints stay in `large_artifacts/qwen_numeric_copy_compiler/checkpoints/`.

## Log

### 2026-06-21

- Created standalone experiment directory and external checkpoint path.
- Implemented the numeric-copy harness and analyzer.
- Tiny frozen-backbone smoke test passed as `runs/smoke_tiny_frozen_copy_trace/`.
- Qwen frozen micro-smoke passed as `runs/smoke_qwen3_4b_frozen_copy_trace/`.
- Frozen numeric-copy pilot completed as `runs/pilot_qwen3_4b_frozen_numeric_copy_trace_mixed_l12/`.
  - Final standard executor accuracy: L4 83.6%, L8 46.1%, L12 16.4%, L24 1.6%.
  - Final paraphrase executor accuracy: L4 87.5%, L8 68.8%, L12 46.1%, L24 14.8%.
  - Final copied argument accuracy stayed high through paraphrase L12 at 93.7%, but standard long-chain argument accuracy was lower.
  - The frozen pilot gate passed, so the live QLoRA numeric-copy condition was run next.
- QLoRA numeric-copy trace condition completed as `runs/main_qwen3_4b_qlora_numeric_copy_trace_mixed_l12/`.
  - Final standard executor accuracy: L4 89.8%, L8 72.7%, L12 46.9%, L24 20.3%.
  - Final paraphrase executor accuracy: L4 85.9%, L8 63.3%, L12 46.1%, L24 5.5%.
  - Standard L24 copied argument accuracy reached 92.9%, up from 77.6% in the frozen pilot.
  - Final-answer controls were run after this condition.
- Direct final-answer QLoRA control completed as `runs/control_qwen3_4b_direct_numeric_copy_distribution_l12/`.
  - Final accuracy stayed near chance on every split.
- Answer-only numeric-copy compiler control completed as `runs/control_qwen3_4b_qlora_numeric_copy_answer_only_l12/`.
  - It solved length-4 programs exactly, but stayed near chance at length 8 and above.
  - This indicates final-answer loss can discover a short numeric-copy shortcut, but did not discover the length-generalizing slot interface under this budget.
- Regenerated aggregate analysis and checkpoint manifest.
- Wrote standalone Markdown and HTML reports.
- Completion audit passed:
  - Source compilation passed.
  - Analyzer regeneration passed.
  - Standalone wording scan found no stale references.
  - Expected README, report, log, analysis, run, source, and manifest files are present.
  - `checkpoint_manifest.csv` indexes 18 large checkpoint files.
