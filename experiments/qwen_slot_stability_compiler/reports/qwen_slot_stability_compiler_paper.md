# Qwen Slot-Stability Numeric-Copy Compiler

## Abstract

This experiment tests whether a QLoRA-adapted `Qwen/Qwen3-4B` model can expose a stable executable program interface for modular arithmetic prompts. Each prompt describes an initial value and a sequence of add, subtract, and multiply updates modulo 97. A trainable compiler reads Qwen hidden states, selects the token positions for the initial value, operations, and arguments, copies exact symbols from deterministic token maps, and executes the copied program with an invisible modular runtime.

The main intervention is paired paraphrase stability. Training batches contain two renderings of the same underlying program, and the stability condition penalizes disagreement between the copied slot distributions for the paired renderings. The matched control receives the same paired data, trace supervision, executor loss, LoRA rank, batch schedule, training seed, and evaluation splits, but no stability loss.

The result is mixed. Trace-supervised numeric-copy compilation works: both compiler arms solve short and medium chains far above chance, while final-answer-only QLoRA remains at chance. The paired stability loss is not a broad win. At 600 steps it improves some longer exact-execution splits, including standard length 12 and paired length 24, but it ties or trails the matched control on other splits and does not produce uniformly higher paired consistency.

## Setup

- Base model: `Qwen/Qwen3-4B`
- Loader: `AutoModelForCausalLM`
- Quantization: 4-bit NF4
- Trainable update: LoRA rank 8, alpha 16, dropout 0.05, target `all-linear`
- Trainable LoRA parameters: 16,515,072
- Compiler head width: 768
- Task: modular arithmetic programs modulo 97
- Train lengths: 1-12 update steps
- Eval lengths: 4, 8, 12, and 24 update steps
- Eval templates: standard, paraphrase, and paired standard/paraphrase renderings of the same program
- Eval size: 64 examples per unpaired split; 64 program pairs per paired split
- Hardware: NVIDIA RTX 6000 Ada Generation, 48 GB class VRAM
- Large checkpoints: `large_artifacts/qwen_slot_stability_compiler/checkpoints/`

The compiler predicts:

- the token position of the initial value;
- ordered operation token positions;
- ordered argument token positions.

Values and operations are copied from per-token maps rather than classified from hidden states alone. The copied program is executed exactly modulo 97 for accuracy. During training, a differentiable executor provides final-answer loss, and trace/selection losses supervise the ordered slot interface.

## Conditions

| Run | Variant | Purpose |
|---|---|---|
| `main_qwen3_4b_qlora_slot_stability_mixed_l12_s600` | `copy_trace_stability` | Numeric-copy compiler with trace loss, executor loss, and paired stability loss. |
| `control_qwen3_4b_qlora_paired_no_stability_mixed_l12_s600` | `copy_trace` | Matched numeric-copy compiler without the stability loss. |
| `control_qwen3_4b_qlora_answer_only_mixed_l12_s600` | `direct` | Final-answer-only QLoRA control without compiler structure. |

The two compiler arms use the same data seed and the same explicit training seed (`2148`). Their step-300 metrics reproduce exactly across the 300-step pilot and the 600-step run, so the final comparison is controlled for initialization and batch order.

## Exact Execution Results

| Split | Stability Compiler | No-Stability Compiler | Answer-Only QLoRA |
|---|---:|---:|---:|
| Standard L4 | 90.6% | 85.9% | 0.0% |
| Standard L8 | 59.4% | 60.9% | 0.0% |
| Standard L12 | 43.8% | 34.4% | 1.6% |
| Standard L24 | 0.0% | 6.2% | 0.0% |
| Paraphrase L4 | 79.7% | 79.7% | 1.6% |
| Paraphrase L8 | 59.4% | 59.4% | 1.6% |
| Paraphrase L12 | 56.2% | 57.8% | 0.0% |
| Paraphrase L24 | 23.4% | 15.6% | 1.6% |
| Paired L4 | 86.7% | 85.9% | 0.0% |
| Paired L8 | 57.8% | 60.2% | 1.6% |
| Paired L12 | 58.6% | 57.8% | 0.0% |
| Paired L24 | 22.7% | 17.2% | 1.6% |

The direct answer-only control stays at chance across all lengths and prompt modes. The compiler arms are therefore not merely benefiting from generic QLoRA adaptation; the trace-supervised symbolic interface is doing the work.

## Paired Consistency

Paired splits render the same latent program twice, once in the standard template and once in a paraphrased template. `Answer consistency` measures whether the two renderings produce the same executed answer. `Both correct` measures whether both renderings are exactly correct. `Program consistency` measures whether the compiled initial value, operation sequence, and argument sequence agree across the pair.

| Split | Stability Answer Consistency | Control Answer Consistency | Stability Both Correct | Control Both Correct | Stability Program Consistency | Control Program Consistency |
|---|---:|---:|---:|---:|---:|---:|
| Paired L4 | 96.9% | 93.8% | 85.9% | 84.4% | 96.9% | 93.8% |
| Paired L8 | 75.0% | 87.5% | 53.1% | 57.8% | 75.0% | 87.5% |
| Paired L12 | 68.8% | 68.8% | 53.1% | 53.1% | 68.8% | 68.8% |
| Paired L24 | 15.6% | 14.1% | 9.4% | 7.8% | 12.5% | 14.1% |

The stability penalty does not reliably increase paired consistency. It helps L4 consistency and slightly improves L24 answer consistency, but the no-stability control is much better at L8 consistency and ties L12. The most defensible interpretation is that paired data plus trace supervision already induces a fairly stable compiler; the explicit KL penalty is only a weak additional bias.

## Diagnostics

The compiler arms learn the easy parts almost perfectly:

- initial value accuracy is 100.0% on every main compiler split;
- operation accuracy is at or near 100.0% on trained and medium lengths;
- operation position accuracy is at or near 100.0% except at length 24;
- argument accuracy remains high per step, but exact program execution compounds the remaining errors.

This is a compounding-error regime. A 90-95% per-step argument extractor can still fail often on length-24 exact execution because every copied argument must be correct.

## Training Dynamics

At step 300, the stability arm showed a large paired length-12 consistency advantage:

| Split | Stability Exec | Control Exec | Stability Pair Consistency | Control Pair Consistency |
|---|---:|---:|---:|---:|
| Paired L12, step 300 | 30.5% | 20.3% | 35.9% | 4.7% |

By step 600, that advantage mostly disappeared on paired L12, while both compiler arms improved. This suggests the stability objective may shape mid-training behavior, but under this budget the matched trace compiler catches up on the core paired L12 interface.

## Interpretation

The positive result is the Qwen-attached compiler itself. With only 600 QLoRA steps, the model and compiler learn an invisible executable interface that solves many modular programs. The final-answer-only QLoRA control does not learn the task at all under the same step budget.

The negative result is the stability loss as a central mechanism. It does not produce a clean monotonic improvement over a matched paired-data trace compiler. The strongest single gain is paraphrase length 24 exact execution, where stability reaches 23.4% versus 15.6%. The strongest loss is paired length 8 consistency, where stability reaches 75.0% versus 87.5%.

The result supports a narrow claim:

1. Qwen3-4B hidden states can be adapted to feed an exact symbolic executor through a learned numeric-copy compiler.
2. Trace supervision is far more effective than final-answer-only QLoRA for this task.
3. Paired paraphrase data is useful, but the tested symmetric-KL stability penalty is not a decisive standalone improvement.

It does not support a claim of broad posttraining intelligence amplification. It shows a practical way to attach an executable latent tool interface to a local 4B-class model for a narrow serial symbolic workload.

## Recommended Follow-Up

The next intervention should target long-chain compounding errors directly rather than adding another agreement penalty. The most promising design is a length curriculum with validation-selected checkpoints:

- train with dense trace supervision and paired templates as in this experiment;
- oversample lengths 12-24 after the compiler reaches high slot accuracy on lengths 4-8;
- add a per-step executor consistency loss that compares intermediate states, not just final answers;
- select by paired length-24 `program_exact`, not by training loss.

Success should require a clear paired length-24 gain over the no-stability compiler, not only a short-chain gain.

## Artifacts

Small files:

- Source: `experiments/qwen_slot_stability_compiler/src/`
- Runs: `experiments/qwen_slot_stability_compiler/runs/`
- Analysis: `experiments/qwen_slot_stability_compiler/analysis/`
- Report: `experiments/qwen_slot_stability_compiler/reports/qwen_slot_stability_compiler_paper.md`
- Manifest: `experiments/qwen_slot_stability_compiler/checkpoint_manifest.csv`

Large files:

- Stability adapter and heads: `large_artifacts/qwen_slot_stability_compiler/checkpoints/main_qwen3_4b_qlora_slot_stability_mixed_l12_s600/`
- Matched no-stability adapter and heads: `large_artifacts/qwen_slot_stability_compiler/checkpoints/control_qwen3_4b_qlora_paired_no_stability_mixed_l12_s600/`
- Answer-only adapter and heads: `large_artifacts/qwen_slot_stability_compiler/checkpoints/control_qwen3_4b_qlora_answer_only_mixed_l12_s600/`
