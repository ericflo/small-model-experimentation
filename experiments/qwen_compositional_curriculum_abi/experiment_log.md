# Qwen Compositional Curriculum ABI Log

## Setup

- Created fresh experiment directory: `/workspace/experiments/qwen_compositional_curriculum_abi`
- Large artifacts directory: `/workspace/large_artifacts/qwen_compositional_curriculum_abi`
- Core question: whether adding depth-2 and depth-3 supervised procedure examples reduces valid-but-wrong errors at held-out deeper depths under constrained ABI decoding.
- Report format: standalone Markdown and HTML with plots.
## Run `smoke_v2`

- Started: 2026-06-26 18:29:58 UTC
- Suite: `smoke`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101`
- Training targets: `atomic_d1,mix_d1_d2,mix_d1_d2_d3`
- Training examples per seed: `18`
- Eval examples per split: `2`
- Eval splits: `eval_indist_d1,eval_comp_d6,eval_template_d6`
- Steps: `1`
- Resample attempts: `2`

Completed `smoke_v2` in 566.9s.

- Metric rows: 30
- Detail rows: 60
- Training log rows: 3

## Run `pilot_v1`

- Started: 2026-06-26 18:40:18 UTC
- Suite: `pilot`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101`
- Training targets: `atomic_d1,mix_d1_d2,mix_d1_d2_d3`
- Training examples per seed: `90`
- Eval examples per split: `4`
- Eval splits: `eval_indist_d1,eval_comp_d2,eval_comp_d3,eval_comp_d4,eval_comp_d6,eval_comp_d8,eval_template_d6,eval_template_d8`
- Steps: `12`
- Resample attempts: `2`

Completed `pilot_v1` in 952.1s.

- Metric rows: 80
- Detail rows: 320
- Training log rows: 15
## Run `main_v1`

- Started: 2026-06-26 18:57:40 UTC
- Suite: `main`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101,202,303`
- Training targets: `atomic_d1,mix_d1_d2,mix_d1_d2_d3`
- Training examples per seed: `120`
- Eval examples per split: `6`
- Eval splits: `eval_indist_d1,eval_comp_d2,eval_comp_d3,eval_comp_d4,eval_comp_d6,eval_comp_d8,eval_template_d6,eval_template_d8`
- Steps: `16`
- Resample attempts: `2`

Completed `main_v1` in 4097.4s.

- Metric rows: 240
- Detail rows: 1440
- Training log rows: 45

