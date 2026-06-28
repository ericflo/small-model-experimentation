# Qwen Numeric-Copy Compiler

## Summary

This experiment tested whether a Qwen causal language model can expose a modular-arithmetic program through learned token roles while exact numeric and operation symbols are copied from a lexical token map. The frozen pilot trained only compiler heads over frozen hidden states. The full condition trained QLoRA adapters and compiler heads live. Both used the same executable modular runtime.

The result is positive. The full QLoRA trace condition reached 89.8% exact execution at standard length 4, 72.7% at length 8, 46.9% at length 12, and 20.3% at length 24. Direct final-answer QLoRA stayed at chance. An answer-only numeric-copy compiler learned length-4 programs exactly but did not generalize to longer chains.

The key finding is that exact numeric copying changes the failure mode. The model no longer has to infer numeric values from hidden states; it only has to learn where the program tokens are. That makes the latent compiler much more reliable, especially on trained-length and moderately longer programs.

## Setup

- Base model: `Qwen/Qwen3-4B`
- Quantization: 4-bit NF4
- Full-condition trainable update: LoRA rank 8, alpha 16, dropout 0.05, target `all-linear`
- Task: modular arithmetic instructions modulo 97
- Train lengths: 1-12 steps
- Evaluation lengths: 4, 8, 12, 24 steps
- Evaluation templates: standard and paraphrase
- Evaluation size: 128 examples per split
- Large checkpoints: `large_artifacts/qwen_numeric_copy_compiler/checkpoints/`

Each prompt describes an initial value and a sequence of add, subtract, and multiply updates. The compiler predicts:

- the token position of the initial value;
- ordered operation token positions;
- ordered argument token positions.

Numeric residues and operation IDs are then copied from deterministic per-token maps. The executor applies the copied program exactly modulo 97.

## Runs

| Run | Variant | Purpose |
|---|---|---|
| `pilot_qwen3_4b_frozen_numeric_copy_trace_mixed_l12` | `copy_trace` | Frozen-backbone pilot gate with trace and executor loss. |
| `main_qwen3_4b_qlora_numeric_copy_trace_mixed_l12` | `copy_trace` | Full QLoRA numeric-copy compiler condition. |
| `control_qwen3_4b_direct_numeric_copy_distribution_l12` | `direct` | Final-answer QLoRA control without compiler structure. |
| `control_qwen3_4b_qlora_numeric_copy_answer_only_l12` | `copy_answer_only` | Numeric-copy compiler trained only from final-answer loss. |

Smoke runs verify the harness and are not part of the main comparison.

## Main QLoRA Result

| Split | Executor | Init | Op | Arg | Arg Pos | Program Exact |
|---|---:|---:|---:|---:|---:|---:|
| Standard L4 | 89.8% | 100.0% | 100.0% | 97.5% | 96.9% | 89.8% |
| Standard L8 | 72.7% | 100.0% | 100.0% | 96.1% | 95.7% | 72.7% |
| Standard L12 | 46.9% | 100.0% | 100.0% | 94.1% | 94.0% | 46.9% |
| Standard L24 | 20.3% | 100.0% | 99.9% | 92.9% | 95.3% | 18.0% |
| Paraphrase L4 | 85.9% | 100.0% | 100.0% | 96.5% | 95.1% | 85.9% |
| Paraphrase L8 | 63.3% | 100.0% | 100.0% | 94.5% | 93.6% | 63.3% |
| Paraphrase L12 | 46.1% | 100.0% | 100.0% | 93.6% | 92.6% | 46.1% |
| Paraphrase L24 | 5.5% | 100.0% | 97.4% | 85.8% | 84.9% | 5.5% |

The compiler solves initial values and operations almost completely. The remaining loss comes from argument slot errors, especially on paraphrase length 24. The standard length-24 result is notable: argument value accuracy stays at 92.9% and program exact reaches 18.0%, even though evaluation length is twice the maximum training length.

## Frozen Pilot

| Split | Executor | Init | Op | Arg | Arg Pos | Program Exact |
|---|---:|---:|---:|---:|---:|---:|
| Standard L4 | 83.6% | 100.0% | 100.0% | 95.5% | 94.5% | 82.8% |
| Standard L8 | 46.1% | 100.0% | 100.0% | 90.2% | 88.5% | 45.3% |
| Standard L12 | 16.4% | 100.0% | 100.0% | 86.5% | 84.2% | 16.4% |
| Standard L24 | 1.6% | 100.0% | 93.9% | 77.6% | 75.3% | 0.0% |
| Paraphrase L4 | 87.5% | 100.0% | 100.0% | 96.9% | 94.1% | 87.5% |
| Paraphrase L8 | 68.8% | 100.0% | 100.0% | 95.4% | 91.5% | 68.8% |
| Paraphrase L12 | 46.1% | 100.0% | 100.0% | 93.7% | 90.4% | 45.3% |
| Paraphrase L24 | 14.8% | 100.0% | 99.5% | 90.2% | 88.2% | 13.3% |

The frozen pilot passed the gate. It showed that Qwen hidden states already support a numeric-copy compiler when trained with token-role traces. The QLoRA condition improves standard long-chain performance but does not uniformly improve every paraphrase split.

## Controls

| Run | Standard L4 | Standard L8 | Standard L12 | Standard L24 | Paraphrase L4 | Paraphrase L8 | Paraphrase L12 | Paraphrase L24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct final-answer QLoRA | 3.1% | 0.8% | 0.0% | 0.8% | 1.6% | 3.1% | 3.9% | 1.6% |
| Answer-only numeric-copy compiler | 100.0% | 0.0% | 1.6% | 2.3% | 100.0% | 7.0% | 1.6% | 0.0% |

The direct control stayed at chance. The answer-only compiler control is more interesting: it solved length-4 programs perfectly but did not extend to longer chains. Its slot-position diagnostics were poor despite perfect length-4 execution, indicating that it found a short-horizon solution rather than the ordered slot interface learned by trace supervision.

## Interpretation

The experiment isolates a concrete bottleneck. A compiler that classifies numeric values from hidden states is too noisy for exact multi-step execution. A compiler that copies numeric values from selected tokens is much stronger. Once numeric copying is used, performance tracks slot localization accuracy.

The result supports three claims:

1. Frozen Qwen hidden states contain enough information for a trace-supervised numeric-copy compiler.
2. Live QLoRA training can improve the compiler on standard longer chains.
3. Final-answer training alone can find a short numeric-copy solution, but it does not discover a length-generalizing ordered compiler under this budget.

The result does not show a universal intelligence improvement. It shows a practical recipe for this class of serial symbolic tasks: learn where the program tokens are, copy exact symbols, and run an invisible executor.

## Recommended Next Experiment

The next experiment should target robust slot generalization rather than numeric value decoding.

Recommended design:

1. Keep numeric-copy readout.
2. Train a slot-stability objective that penalizes drift in ordered operation and argument slots across paraphrase variants of the same program.
3. Add a curriculum that mixes short and long lengths every batch instead of relying on uniform random lengths.
4. Save mid-run checkpoints and select by validation program exactness, because the frozen pilot showed different splits peaking at different times.

Success criteria:

- standard and paraphrase L12 program exact above 60%;
- paraphrase L24 above 20%;
- answer-only control remains unable to solve L8+ without traces.

## Artifacts

Small files:

- Code: `experiments/qwen_numeric_copy_compiler/src/`
- Runs: `experiments/qwen_numeric_copy_compiler/runs/`
- Analysis: `experiments/qwen_numeric_copy_compiler/analysis/`
- Manifest: `experiments/qwen_numeric_copy_compiler/checkpoint_manifest.csv`

Large files:

- Frozen pilot heads: `large_artifacts/qwen_numeric_copy_compiler/checkpoints/pilot_qwen3_4b_frozen_numeric_copy_trace_mixed_l12/`
- Main QLoRA adapter and heads: `large_artifacts/qwen_numeric_copy_compiler/checkpoints/main_qwen3_4b_qlora_numeric_copy_trace_mixed_l12/`
- Direct control adapter and heads: `large_artifacts/qwen_numeric_copy_compiler/checkpoints/control_qwen3_4b_direct_numeric_copy_distribution_l12/`
- Answer-only control adapter and heads: `large_artifacts/qwen_numeric_copy_compiler/checkpoints/control_qwen3_4b_qlora_numeric_copy_answer_only_l12/`
