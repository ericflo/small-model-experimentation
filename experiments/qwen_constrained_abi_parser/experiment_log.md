# Qwen Constrained ABI Parser Log

## Setup

- Created: 2026-06-26 UTC
- Fresh experiment directory: `experiments/qwen_constrained_abi_parser`
- Large artifact directory: `large_artifacts/qwen_constrained_abi_parser`
- Base model: `Qwen/Qwen3-4B`
- Primary question: can a grammar-constrained stack ABI decoder and a
  canonical parse stage improve executable procedure accuracy, not merely
  valid-program rate?
- Primary readouts: execution accuracy by composition depth, template-shift
  execution, validity versus execution, correct-given-valid, parse exactness,
  decoder divergence, and failure taxonomy.

## Run `smoke_v1`

- Started: 2026-06-26 15:43:38 UTC
- Suite: `smoke`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101`
- Training targets: `answer_only,program_stack,parse_emit`
- Training examples per seed: `18`
- Eval examples per split: `3`
- Steps: `1`
- Resample attempts: `2`

Completed `smoke_v1` in 704.5s.

- Metric rows: 64
- Detail rows: 192
- Training log rows: 3

## Run `pilot_v1`

- Started: 2026-06-26 15:56:19 UTC
- Suite: `pilot`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101`
- Training targets: `answer_only,program_stack,parse_emit`
- Training examples per seed: `120`
- Eval examples per split: `6`
- Steps: `12`
- Resample attempts: `2`

Completed `pilot_v1` in 714.1s.

- Metric rows: 64
- Detail rows: 384
- Training log rows: 15

## Run `main_v1`

- Started: 2026-06-26 16:08:48 UTC
- Suite: `main`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101,202,303,404,505`
- Training targets: `answer_only,program_stack,parse_emit`
- Training examples per seed: `180`
- Eval examples per split: `8`
- Steps: `24`
- Resample attempts: `2`

Completed `main_v1` in 4058.7s.

- Metric rows: 320
- Detail rows: 2560
- Training log rows: 75

