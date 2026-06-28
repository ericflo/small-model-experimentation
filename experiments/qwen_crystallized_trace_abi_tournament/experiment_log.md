# Qwen Crystallized Trace ABI Tournament Log

## Setup

- Created: 2026-06-26 UTC
- Fresh experiment directory: `experiments/qwen_crystallized_trace_abi_tournament`
- Large artifact directory: `large_artifacts/qwen_crystallized_trace_abi_tournament`
- Base model: `Qwen/Qwen3-4B`
- Primary question: can dense trace supervision over a practical executable
  ABI improve held-out exact-answer accuracy over answer-only fine-tuning?

## Run `smoke_v1`

- Started: 2026-06-26 04:20:46 UTC
- Suite: `smoke`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101`
- Arms: `answer,python`
- Train examples per seed: `24`
- Eval examples per split: `8`
- Steps: `2`

Completed `smoke_v1` in 146.4s.

- Metric rows: 9
- Detail rows: 72
- Training log rows: 4

## Run `pilot_v1`

- Started: 2026-06-26 04:24:02 UTC
- Suite: `pilot`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101`
- Arms: `answer,python,json,stack`
- Train examples per seed: `96`
- Eval examples per split: `16`
- Steps: `24`

Completed `pilot_v1` in 367.7s.

- Metric rows: 15
- Detail rows: 240
- Training log rows: 28

## Run `main_v1`

- Started: 2026-06-26 04:30:45 UTC
- Suite: `main`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101,202`
- Arms: `answer,python,json,stack`
- Train examples per seed: `192`
- Eval examples per split: `24`
- Steps: `48`

Completed `main_v1` in 1324.1s.

- Metric rows: 27
- Detail rows: 648
- Training log rows: 56

