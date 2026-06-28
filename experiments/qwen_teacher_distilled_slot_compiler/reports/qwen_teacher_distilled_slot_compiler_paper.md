# Teacher-Distilled Slot Compiler for Qwen Numeric Programs

## Abstract

This experiment tests whether oracle slot-teacher losses improve a QLoRA-adapted `Qwen/Qwen3-4B` compiler on synthetic modular-arithmetic programs. The compiler reads the full prompt hidden sequence, predicts executable slots, copies numbers and operations, and executes the copied program modulo 97. The teacher variants add auxiliary supervision for slot localization, with one arm also matching oracle token representations.

The result is negative. On fresh length-24 paired standard/paraphrase programs, the matched no-teacher control scored 27.1% exact execution with 72.7% paired compiler-state consistency. The heavier teacher arm scored 27.9% exact execution but only 55.5% consistency. The lower-weight soft-position-only arm scored 18.4% exact execution and 41.0% consistency. Oracle slot imitation did not improve the selected checkpoint and often made paraphrase-invariant compilation worse.

## Question

Can a Qwen-attached numeric compiler bind executable slots more reliably when ordinary trace and state supervision are augmented with oracle slot-teacher losses?

The hypothesis was that a substantial part of the remaining error came from misbinding surface tokens to executable slots. If true, teacher losses should improve long-chain execution and standard/paraphrase consistency, especially on length-24 programs where one copied-step error usually corrupts the final answer.

## Method

All main runs used `Qwen/Qwen3-4B` with 4-bit QLoRA adapters, a copied-symbol program compiler head, a deterministic modular executor, and the same synthetic program curriculum:

| Stage | Length range | Steps |
|---|---:|---:|
| short | 1-4 | 200 |
| medium | 1-8 | 200 |
| train | 1-12 | 200 |
| long | 8-24 | 300 |

The model was trained with paired standard/paraphrase examples and selected by highest validation `paired_len24_executor_accuracy`. Large adapter and head checkpoints are stored outside the experiment directory under `large_artifacts/qwen_teacher_distilled_slot_compiler/checkpoints/`.

Three main arms were run:

| Run | Teacher position weight | Teacher representation weight | Description |
|---|---:|---:|---|
| `main_control_light_state_s900` | 0.00 | 0.00 | Matched light-state compiler control |
| `main_teacher_slot_distill_s900` | 0.10 | 0.05 | Soft local position targets plus oracle slot-representation matching |
| `main_teacher_softpos_low_s900` | 0.03 | 0.00 | Lower-weight soft position targets only |

The primary held-out retest used 256 fresh length-24 programs per single-rendering split and 256 paired programs rendered in both standard and paraphrased forms, giving 512 examples in the paired row.

## Results

### Selected Validation Checkpoints

| Run | Selected step | Paired L24 exact | Paired state consistency | Standard L24 exact | Paraphrase L24 exact |
|---|---:|---:|---:|---:|---:|
| Control | 800 | 30.5% | 67.2% | 37.5% | 25.0% |
| Teacher position + representation | 800 | 28.1% | 59.4% | 37.5% | 21.9% |
| Low soft-position only | 800 | 23.4% | 32.8% | 29.7% | 15.6% |

The matched control was the best validation-selected arm. Both teacher variants selected the same training step but had lower paired exact execution and lower paired compiler-state consistency.

### Fresh Selected-Checkpoint Retest

| Run | Fresh standard L24 exact | Fresh paraphrase L24 exact | Fresh paired L24 exact | Fresh paired state consistency |
|---|---:|---:|---:|---:|
| Control | 30.5% | 27.7% | 27.1% | 72.7% |
| Teacher position + representation | 30.5% | 28.1% | 27.9% | 55.5% |
| Low soft-position only | 27.0% | 11.7% | 18.4% | 41.0% |

The heavier teacher arm had a tiny fresh paired exact-execution edge over the control, 27.9% versus 27.1%, but this is not a useful win: it came with a large consistency loss, and its selected validation metric was lower. The low-weight position-only arm was clearly worse on the fresh retest, especially on paraphrased prompts.

## Interpretation

The teacher losses did not address the dominant bottleneck. The compiler already achieved near-perfect initialization and operation localization, and argument localization stayed around 93-95% at length 24 in the main arms. The remaining failure mode is therefore not simply "the model cannot find the right token." Small residual copy errors compound across 24 steps, and the paired consistency metrics show that the learned compiler is still sensitive to surface rendering.

The representation-matching teacher likely over-constrained the compiler to prompt-surface-specific hidden vectors. That explains why the heavier teacher arm tied answer accuracy but lost paired compiler-state consistency. The low-weight soft-position-only arm briefly improved mid-training transfer at step 600, but the effect did not survive checkpoint selection or fresh retesting.

## Conclusion

Oracle slot imitation is not the highest-leverage next direction for this Qwen-attached compiler. It can provide a transient curriculum signal, but it does not improve the best selected checkpoint and can damage paraphrase-invariant compilation.

The most impactful next experiment should move from slot imitation to execution-level pressure:

1. Train the compiler with verifier-guided self-correction or search over copied slots, so the model learns from execution failures rather than only from local token labels.
2. Add a differentiable or sampled repair loop that proposes alternate arguments for low-confidence slots and backpropagates or reinforces the execution result.
3. Use a short policy-gradient stage over compiled programs with a strong supervised anchor, rewarding exact final execution and paired standard/paraphrase program agreement.

The practical recommendation is to start with verifier-guided slot repair, because it directly targets the observed failure mode: high per-slot accuracy that still collapses under long-chain composition.
