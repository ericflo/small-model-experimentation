# Qwen LoRA Parser Compiler Experiment Log

## Objective

Test whether live QLoRA updates can make a small causal language model expose a clean modular-arithmetic program in its hidden states, and whether a learned compiler/executor can use that program to answer longer arithmetic chains.

## Experiment Question

Can a LoRA-adapted model learn token-level roles and program symbols for synthetic arithmetic instructions well enough to support a compiled executor, including on paraphrased prompts and longer held-out chains?

## Planned Gates

1. Token-tagger pilot: train QLoRA plus parser heads on init, operation, argument, and token-position supervision.
2. Parser/executor condition: add differentiable execution loss on the predicted modular program.
3. Answer-only control: train the same QLoRA substrate without trace supervision.

## Primary Metrics

- `init_pos_accuracy`, `op_pos_accuracy`, `arg_pos_accuracy`: whether the parser attends to the intended source tokens.
- `init_accuracy`, `op_accuracy`, `arg_accuracy`: whether those tokens are decoded into correct program symbols.
- `program_exact`: whether the whole predicted program is exact.
- `executor_accuracy`: whether compiled execution returns the correct answer.
- Standard and paraphrase splits are evaluated separately at multiple chain lengths.

## Artifact Policy

Lightweight outputs stay in `experiments/qwen_lora_parser_compiler/runs/`.

Large adapter and head checkpoints stay in `large_artifacts/qwen_lora_parser_compiler/checkpoints/`.

## Log

### 2026-06-21

- Created standalone experiment directory and large-artifact checkpoint path.
- Added a live QLoRA training harness with a token-tagger pilot variant and parser/executor variants.
- Tightened the synthetic generator so operation-position labels point at the operation token rather than the end of the surrounding phrase.
- Tiny random LoRA smoke test passed and wrote `runs/smoke_tiny_tagger/results.json`.
- Qwen 3 4B 4-bit LoRA micro-smoke passed and wrote `runs/smoke_qwen3_4b_qlora_tagger/results.json`.
- Qwen QLoRA token-tagger pilot completed as `runs/pilot_qwen3_4b_qlora_tagger_mixed_l12/`.
  - Operation symbols were mostly solved: 88.2-99.2% across final splits.
  - Argument positions were mostly found: 77.4-94.5%.
  - Argument values remained the limiting error: 65.4-77.7%.
- Qwen QLoRA trace+executor condition completed as `runs/main_qwen3_4b_qlora_trace_argstrong_mixed_l12/`.
  - Short-chain executor accuracy improved over the tagger-only pilot.
  - Standard length-4 reached 32.8%; paraphrase length-4 reached 40.6%.
  - Length-8 remained weak and length-12/24 remained near zero.
  - Argument value accuracy remained the main bottleneck.
- Direct final-answer QLoRA control completed as `runs/control_qwen3_4b_direct_mixed_l12/`.
  - Final accuracy stayed near chance on every split.
- Answer-only compiler control completed as `runs/control_qwen3_4b_qlora_answer_only_mixed_l12/`.
  - Final executor accuracy stayed near chance and the parser did not discover useful symbols.
- Regenerated aggregate analysis and checkpoint manifest.
- Wrote standalone Markdown and HTML reports.
- Recommended next step: a numeric-copy compiler variant that directly targets the argument-value bottleneck.
