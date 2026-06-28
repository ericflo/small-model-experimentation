# Qwen Trace Procedure Depth Stress Log

## Setup

- Created: 2026-06-26 UTC
- Fresh experiment directory: `experiments/qwen_trace_procedure_depth_stress`
- Large artifact directory: `large_artifacts/qwen_trace_procedure_depth_stress`
- Base model: `Qwen/Qwen3-4B`
- Primary question: can a model trained on atomic procedures compose known
  primitives into deeper executable procedures when the generated procedure is
  run by a deterministic interpreter?
- Primary readouts: execution accuracy by composition depth, template-shift
  execution by depth, final-answer-vs-execution gap, and failure taxonomy.

## Run `smoke_v1`

- Started: 2026-06-26 07:14:05 UTC
- Suite: `smoke`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101`
- Arms: `answer_only,trace_stack_final,trace_stack_no_final,program_stack`
- Training examples per seed: `24`
- Eval examples per split: `6`
- Steps: `2`

Completed `smoke_v1` in 657.7s.

- Metric rows: 32
- Detail rows: 192
- Training log rows: 8

## Run `pilot_v1`

- Started: 2026-06-26 07:25:33 UTC
- Suite: `pilot`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101`
- Arms: `answer_only,trace_stack_final,trace_stack_no_final,program_stack`
- Training examples per seed: `120`
- Eval examples per split: `12`
- Steps: `20`

Completed `pilot_v1` in 533.6s.

- Metric rows: 32
- Detail rows: 384
- Training log rows: 20

## Run `main_v1`

- Started: 2026-06-26 07:35:04 UTC
- Suite: `main`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101,202,303,404,505`
- Arms: `answer_only,trace_stack_final,trace_stack_no_final,program_stack`
- Training examples per seed: `180`
- Eval examples per split: `12`
- Steps: `32`

Completed `main_v1` in 2743.2s.

- Metric rows: 160
- Detail rows: 1920
- Training log rows: 100

