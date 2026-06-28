# Qwen Program-Only Executable ABI Log

## Setup

- Created: 2026-06-26 UTC
- Fresh experiment directory: `experiments/qwen_program_only_executable_abi`
- Large artifact directory: `large_artifacts/qwen_program_only_executable_abi`
- Base model: `Qwen/Qwen3-4B`
- Primary question: can a model emit an executable program when the final answer
  is not present in the target and the score comes only from interpreting the
  generated program?

## Run `smoke_v1`

- Started: 2026-06-26 06:06:48 UTC
- Suite: `smoke`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101`
- Arms: `answer_only,program_stack`
- Train examples per seed: `24`
- Eval examples per split: `8`
- Steps: `2`

Completed `smoke_v1` in 79.4s.

- Metric rows: 9
- Detail rows: 72
- Training log rows: 4

## Run `pilot_v1`

- Started: 2026-06-26 06:09:13 UTC
- Suite: `pilot`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101`
- Arms: `answer_only,trace_stack_final,program_stack,program_python`
- Train examples per seed: `96`
- Eval examples per split: `16`
- Steps: `24`

Completed `pilot_v1` in 297.7s.

- Metric rows: 15
- Detail rows: 240
- Training log rows: 28

## Run `main_v1`

- Started: 2026-06-26 06:15:31 UTC
- Suite: `main`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101,202`
- Arms: `answer_only,trace_stack_final,program_stack,program_python`
- Train examples per seed: `192`
- Eval examples per split: `24`
- Steps: `48`

Completed `main_v1` in 1177.0s.

- Metric rows: 27
- Detail rows: 648
- Training log rows: 56

