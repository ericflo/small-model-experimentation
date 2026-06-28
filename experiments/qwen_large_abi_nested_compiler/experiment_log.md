# Qwen Large ABI Nested Compiler Log

## Setup

- Created fresh experiment directory: `/workspace/experiments/qwen_large_abi_nested_compiler`
- Large artifacts directory: `/workspace/large_artifacts/qwen_large_abi_nested_compiler`
- Core question: whether constrained ABI compilation survives both larger primitive libraries and nested branch/sub-procedure structure.
- Report format: standalone Markdown and HTML with plots.
## Run `smoke_v1`

- Started: 2026-06-26 23:01:46 UTC
- Suite: `smoke`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101`
- Training targets: `abi32_chain_d3,abi128_chain_d3,abi32_nested_d3,abi128_nested_d3`
- Training examples per seed: `24`
- Eval examples per split: `1`
- Eval splits: `eval_chain_d16,eval_nested_l8,eval_chain_template_d16,eval_nested_template_l8`
- Steps: `1`
- Resample attempts: `3`

Completed `smoke_v1` in 617.7s.

- Metric rows: 40
- Detail rows: 40
- Training log rows: 4

## Run `pilot_v1`

- Started: 2026-06-26 23:12:48 UTC
- Suite: `pilot`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101`
- Training targets: `abi32_chain_d3,abi128_chain_d3,abi32_nested_d3,abi128_nested_d3`
- Training examples per seed: `96`
- Eval examples per split: `3`
- Eval splits: `eval_chain_d3,eval_chain_d8,eval_chain_d16,eval_chain_template_d16,eval_nested_l2,eval_nested_l4,eval_nested_l8,eval_nested_template_l8`
- Steps: `8`
- Resample attempts: `3`

Completed `pilot_v1` in 1514.4s.

- Metric rows: 80
- Detail rows: 240
- Training log rows: 20
- Pilot read:
  - 32-op chain-only constrained chain depth-16 reached 100.0%.
  - 128-op chain-only constrained chain depth-16 also reached 100.0%, but template chain depth-16 fell to 33.3%.
  - Chain-only curricula transferred to 2-branch nested tasks at 66.7% but collapsed at 4 and 8 branches.
  - 32-op nested curriculum produced a nonzero nested-8 row (33.3%) but did not solve nested-4 or template nested-8.
  - 128-op nested curriculum did not improve nested-8 in the pilot.
  - Added `eval_nested_l3` before main so trained nested-boundary performance is measured explicitly.
## Run `main_v1`

- Started: 2026-06-26 23:39:21 UTC
- Suite: `main`
- Model: `Qwen/Qwen3-4B`
- Seeds: `101,202,303`
- Training targets: `abi32_chain_d3,abi128_chain_d3,abi32_nested_d3,abi128_nested_d3`
- Training examples per seed: `240`
- Eval examples per split: `5`
- Eval splits: `eval_chain_d3,eval_chain_d8,eval_chain_d16,eval_chain_template_d16,eval_nested_l2,eval_nested_l3,eval_nested_l4,eval_nested_l8,eval_nested_template_l8`
- Steps: `24`
- Resample attempts: `3`

Completed `main_v1` in 5622.3s.

- Metric rows: 270
- Detail rows: 1350
- Training log rows: 60
- Final read:
  - The 128-operation catalog did not break linear chain compilation under constrained decoding: `abi128_chain_d3` reached 100.0% at chain depth 16, matching `abi32_chain_d3`.
  - Chain-only training did not learn nested branch structure: constrained nested-8 was 6.7% for 32 ops and 13.3% for 128 ops.
  - Shallow nested training transferred strongly beyond the trained branch counts: constrained nested-8 reached 73.3% for 32 ops and 86.7% for 128 ops.
  - Nested depth 4 was solved by both nested curricula at 100.0%, showing stable near-depth transfer.
  - Template-shifted nested-8 remained weak: 46.7% for 32-op nested training and 40.0% for 128-op nested training, even though validity was 100.0%.
  - Gold ABI sanity arms were 100.0% across both ABI sizes and all splits, so residual errors are compiler selection/grounding errors rather than interpreter or decoder plumbing errors.
  - Read: operation selection at 128 ops is not the immediate blocker for linear pipelines; nested/control-flow structure is trainable with shallow examples; wording-robust nested grounding remains the next bottleneck.
