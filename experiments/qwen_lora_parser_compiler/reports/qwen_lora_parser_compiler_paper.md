# Qwen LoRA Parser Compiler

## Summary

This experiment tested whether QLoRA posttraining can install a small symbolic interface on top of `Qwen/Qwen3-4B`. The model was trained live with LoRA adapters while a parser read the final hidden sequence and emitted a modular-arithmetic program: an initial value, ordered operations, and ordered arguments. A differentiable executor then ran the predicted program modulo 97.

The result is mixed. Trace supervision created a real, measurable program interface that ordinary final-answer training did not discover. The full trace+executor condition reached 32.8% standard length-4 executor accuracy and 40.6% paraphrase length-4 accuracy, while direct final-answer QLoRA and answer-only compiler training stayed near chance. The failure mode is also clear: argument value decoding is not accurate enough. It remains near 80% on length-4 programs and falls with length, which compounds into near-zero exact programs by length 12.

The experiment supports a narrow claim: explicit parser supervision can make a small QLoRA-adapted model expose a partial symbolic program in hidden states. It does not support a broad claim of large universal intelligence improvement from this recipe.

## Setup

- Base model: `Qwen/Qwen3-4B`
- Loading: 4-bit NF4 quantization
- Trainable model update: LoRA rank 8, alpha 16, dropout 0.05, target `all-linear`
- Task: synthetic text instructions for arithmetic modulo 97
- Train lengths: 1-12 steps
- Evaluation lengths: 4, 8, 12, 24 steps
- Evaluation templates: standard and paraphrase
- Evaluation size: 64 examples per split
- Large checkpoint storage: `large_artifacts/qwen_lora_parser_compiler/checkpoints/`

The parser predicts:

- token positions for the initial value, each operation, and each argument;
- symbols for the initial value, operation class, and argument value;
- an answer distribution through exact differentiable modular execution.

## Runs

| Run | Variant | Purpose |
|---|---|---|
| `pilot_qwen3_4b_qlora_tagger_mixed_l12` | `qlora_tagger` | Pilot gate: learn token roles and program symbols with trace supervision only. |
| `main_qwen3_4b_qlora_trace_argstrong_mixed_l12` | `qlora_trace` | Full condition: trace supervision plus differentiable executor loss, with stronger argument supervision. |
| `control_qwen3_4b_direct_mixed_l12` | `direct` | Ordinary final-answer QLoRA control using an answer-marker hidden-state classifier. |
| `control_qwen3_4b_qlora_answer_only_mixed_l12` | `qlora_answer_only` | Compiler/executor trained from final-answer loss only, without trace supervision. |

Smoke runs are present only to verify the harness and checkpoint path.

## Main Result

### Full Trace+Executor Condition

| Split | Executor | Init | Op | Arg | Program Exact |
|---|---:|---:|---:|---:|---:|
| Standard L4 | 32.8% | 100.0% | 97.7% | 79.7% | 32.8% |
| Standard L8 | 10.9% | 100.0% | 97.7% | 77.9% | 9.4% |
| Standard L12 | 0.0% | 100.0% | 94.0% | 70.7% | 0.0% |
| Standard L24 | 0.0% | 100.0% | 81.4% | 67.4% | 0.0% |
| Paraphrase L4 | 40.6% | 100.0% | 96.9% | 82.4% | 40.6% |
| Paraphrase L8 | 6.2% | 100.0% | 97.7% | 75.8% | 6.2% |
| Paraphrase L12 | 4.7% | 100.0% | 90.5% | 69.3% | 0.0% |
| Paraphrase L24 | 0.0% | 100.0% | 75.5% | 53.1% | 0.0% |

The parser learned initial values and operations well. It also usually located argument tokens. The blocker is argument value decoding. A per-step argument error rate around 20-45% makes exact execution collapse as program length grows.

### Token-Tagger Pilot

| Split | Executor | Init | Op | Arg | Op Pos | Arg Pos | Program Exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard L4 | 17.2% | 96.9% | 98.0% | 76.2% | 100.0% | 89.5% | 17.2% |
| Standard L8 | 3.1% | 98.4% | 94.9% | 73.8% | 100.0% | 85.2% | 0.0% |
| Standard L12 | 0.0% | 96.9% | 95.6% | 68.8% | 100.0% | 82.4% | 0.0% |
| Standard L24 | 1.6% | 100.0% | 88.2% | 65.4% | 90.2% | 77.4% | 0.0% |
| Paraphrase L4 | 32.8% | 98.4% | 99.2% | 77.7% | 100.0% | 94.5% | 32.8% |
| Paraphrase L8 | 17.2% | 95.3% | 99.0% | 77.5% | 100.0% | 93.0% | 15.6% |
| Paraphrase L12 | 0.0% | 100.0% | 99.2% | 71.6% | 100.0% | 91.7% | 0.0% |
| Paraphrase L24 | 1.6% | 93.8% | 97.0% | 71.2% | 99.4% | 87.3% | 0.0% |

This pilot passed the narrow gate: QLoRA plus trace heads can make Qwen hidden states parseable. It did not pass the stronger gate needed for long exact execution.

### Controls

| Run | Standard L4 | Standard L8 | Standard L12 | Standard L24 | Paraphrase L4 | Paraphrase L8 | Paraphrase L12 | Paraphrase L24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct final-answer QLoRA | 3.1% | 0.0% | 0.0% | 1.6% | 0.0% | 0.0% | 1.6% | 0.0% |
| Answer-only compiler | 0.0% | 1.6% | 3.1% | 3.1% | 0.0% | 1.6% | 0.0% | 0.0% |

The controls stayed at chance. The learned structure in the trace-supervised runs is therefore not explained by ordinary answer fitting, and the compiler interface was not discovered from final-answer loss alone.

## Interpretation

The useful signal is not the final executor score by itself. The useful signal is that explicit trace supervision made internal hidden states reliably expose parts of a program:

- Initial value decoding reached 100.0% in the full condition.
- Operation decoding stayed high through length 12 and degraded at length 24.
- Argument token localization was usually high.
- Argument value decoding was the unresolved bottleneck.

This is an important distinction. The experiment did not fail because the parser could not find the computation. It failed because the numeric argument channel was too noisy for exact multi-step execution. With 8 or 12 steps, even a good but imperfect per-step argument classifier produces very few exact programs.

The result suggests the next experiment should not add more recurrence or a larger executor first. The highest-leverage change is to replace the weak argument-value readout with a stronger numeric interface.

## Recommended Next Experiment

Run a numeric-copy compiler variant:

1. Keep the same QLoRA parser and role supervision.
2. Add an auxiliary numeric-token decoder for argument values, separate from the semantic hidden-state classifier.
3. Add a constrained value head that only predicts valid generated argument ranges for each operation class.
4. Evaluate whether argument accuracy can exceed 95% at length 8 before running longer chains.

Success criterion:

- argument value accuracy above 95% at L8;
- program exact above 50% at L8;
- direct and answer-only controls still near chance.

This is the shortest path to determining whether the symbolic interface is fundamentally useful or merely limited by the current numeric readout.

## Artifacts

Small files:

- Code: `experiments/qwen_lora_parser_compiler/src/`
- Runs: `experiments/qwen_lora_parser_compiler/runs/`
- Analysis: `experiments/qwen_lora_parser_compiler/analysis/`
- Manifest: `experiments/qwen_lora_parser_compiler/checkpoint_manifest.csv`

Large files:

- Pilot adapter and heads: `large_artifacts/qwen_lora_parser_compiler/checkpoints/pilot_qwen3_4b_qlora_tagger_mixed_l12/`
- Full trace+executor adapter and heads: `large_artifacts/qwen_lora_parser_compiler/checkpoints/main_qwen3_4b_qlora_trace_argstrong_mixed_l12/`
- Direct control adapter and heads: `large_artifacts/qwen_lora_parser_compiler/checkpoints/control_qwen3_4b_direct_mixed_l12/`
- Answer-only compiler control adapter and heads: `large_artifacts/qwen_lora_parser_compiler/checkpoints/control_qwen3_4b_qlora_answer_only_mixed_l12/`

