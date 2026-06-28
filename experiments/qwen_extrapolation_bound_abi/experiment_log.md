# Qwen Extrapolation-Bound ABI Log

## Setup

- Created fresh experiment directory: `/workspace/experiments/qwen_extrapolation_bound_abi`
- Large artifacts directory: `/workspace/large_artifacts/qwen_extrapolation_bound_abi`
- Core question: how far constrained ABI compilation extrapolates beyond the maximum supervised composition depth.
- Report format: standalone Markdown and HTML with plots.
## Run `smoke_v1`

- Started: 2026-06-26 20:32:31 UTC
- Suite: `smoke`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101`
- Training targets: `atomic_d1,mix_d1_d2_d3,mix_d1_to_d6,mix_d1_to_d8`
- Training examples per seed: `24`
- Eval examples per split: `2`
- Eval splits: `eval_indist_d1,eval_comp_d16,eval_template_d16`
- Steps: `1`
- Resample attempts: `3`

Completed `smoke_v1` in 462.5s.

- Metric rows: 27
- Detail rows: 54
- Training log rows: 4

## Run `pilot_v1`

- Started: 2026-06-26 20:40:44 UTC
- Suite: `pilot`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101`
- Training targets: `atomic_d1,mix_d1_d2_d3,mix_d1_to_d6,mix_d1_to_d8`
- Training examples per seed: `96`
- Eval examples per split: `3`
- Eval splits: `eval_indist_d1,eval_comp_d3,eval_comp_d6,eval_comp_d8,eval_comp_d12,eval_comp_d16,eval_template_d8,eval_template_d12,eval_template_d16`
- Steps: `8`
- Resample attempts: `3`

Completed `pilot_v1` in 1210.6s.

- Metric rows: 81
- Detail rows: 243
- Training log rows: 20
- Sanity: gold ABI constrained execution and validity were 100% on every pilot split, including depth 16.
- Pilot read: constrained validity stayed at 100%; depth-12/depth-16 execution remained nonzero across curricula, so the main run can measure the extrapolation bound rather than only observing collapse.
## Run `main_v1`

- Started: 2026-06-26 21:01:36 UTC
- Suite: `main`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101,202,303`
- Training targets: `atomic_d1,mix_d1_d2_d3,mix_d1_to_d6,mix_d1_to_d8`
- Training examples per seed: `180`
- Eval examples per split: `6`
- Eval splits: `eval_indist_d1,eval_comp_d3,eval_comp_d6,eval_comp_d8,eval_comp_d12,eval_comp_d16,eval_template_d8,eval_template_d12,eval_template_d16`
- Steps: `20`
- Resample attempts: `3`

Completed `main_v1` in 4804.4s.

- Metric rows: 243
- Detail rows: 1458
- Training log rows: 60
- Main read:
  - Atomic-only constrained depth-16 execution: 61.1% standard, 72.2% template shift.
  - Depth-3 curriculum constrained depth-16 execution: 100.0% standard, 94.4% template shift.
  - Depth-6 curriculum constrained depth-16 execution: 100.0% standard, 88.9% template shift.
  - Depth-8 curriculum constrained depth-16 execution: 88.9% standard, 88.9% template shift.
  - Constrained validity was 100% throughout trained arms, so gains are correct-given-valid composition gains rather than syntax gains.
  - Practical corpus implication: include shallow composed examples through depth 3 first; deeper examples are not automatically beneficial at this budget and may add variance.
