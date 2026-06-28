# Experiment Log

## 2026-06-23

- Created standalone experiment directory with `src/`, `runs/`, `analysis/`, `reports/`, and an external large-artifact checkpoint root.
- Design choice: train live QLoRA adapters through Qwen hidden states rather than only training a frozen feature head.
- Planned conditions:
  - `frozen_trace`: frozen Qwen hidden states with a trained bytecode compiler head.
  - `qlora_trace`: QLoRA Qwen plus compiler head trained on dense executable bytecode traces.
  - `qlora_trace_ei`: QLoRA trace seed training followed by answer-verified expert-iteration target collection.
- Implemented the self-contained experiment harness in `src/qwen_lora_typed_bytecode_trace_compiler_experiment.py`.
- Verified the task generator and VM on 2,000 generated examples across standard and hard settings.
- Ran `smoke_qwen3_4b_qlora_trace`, a two-step live QLoRA smoke test. It loaded `Qwen/Qwen3-4B`, trained 16.5M LoRA parameters plus the bytecode head, wrote metrics, and saved adapter/head checkpoints to the large-artifact root.
- Ran `pilot_qwen3_4b_qlora_trace_s128`. The pilot showed real learning: quick validation bytecode moved above the smoke baseline and full fresh/paraphrase splits showed answer-verified local-search headroom.
- Ran `main_qwen3_4b_qlora_trace_s512`. Fresh paired direct executable bytecode reached 68.0%; answer-verified local search reached 85.9%; hard-composition direct bytecode reached 55.5%.
- Ran `control_qwen3_4b_frozen_trace_s512`. Fresh paired direct executable bytecode reached 66.4%, showing that dense trace supervision over frozen Qwen hidden states is already strong and live QLoRA adds only a modest direct lift here.
- Ran `main_qwen3_4b_qlora_trace_ei_s256_u1024`. Expert iteration improved fresh paired direct bytecode from 21.9% after seed training to 48.4% after two rounds, but did not beat the 512 gold-trace supervised run.
- Ran `control_qwen3_4b_qlora_answer_s512`. Fresh paired answer accuracy was 14.8%, far below executable trace supervision.
- Generated aggregate CSVs, figures, standalone Markdown report, and standalone HTML report with `src/analyze_qwen_lora_typed_bytecode_trace_compiler.py`.
