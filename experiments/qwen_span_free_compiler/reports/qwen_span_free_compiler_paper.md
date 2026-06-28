# Span-Free Qwen Latent Compiler with Alignment Bootstrap

## Abstract

This experiment tests whether a small trainable compiler can read the full
hidden sequence of a frozen Qwen3.5-4B model and configure an executable latent
program without receiving token-span features at inference time. A plain
query-context reader fails to extract numeric program symbols. Token-local
classification improves argument extraction but still fails to locate the
initial value. Adding trace-time attention alignment changes the result: the
compiler reaches 99.6% exact execution on the trained length-4 standard
template, while a direct answer head and an answer-only compiler remain at
chance. Answer-only continuation preserves the installed interface. The result
does not transfer to untrained later step slots or paraphrased prompt wording.

## Setup

The task is modular program execution. Each prompt gives an initial value `x`,
a sequence of add, subtract, and multiply updates modulo 97, and an answer
marker. The model must recover the final value.

The frozen model is used only as a hidden-state source. The trainable compiler
receives:

- the padded full hidden sequence,
- a sequence mask,
- no numeric token span features at inference time.

The compiler predicts:

- initial value,
- per-step operation,
- per-step argument.

Those symbols are executed by a differentiable modular executor during training
and by argmax symbolic execution for exact accuracy.

## Variants

| Variant | Training signal |
|---|---|
| `direct` | answer classification from the answer-marker hidden state |
| `compiler_answer_only` | final-answer loss through the executor |
| `compiler_trace` | symbol trace loss, executor loss, and attention alignment |
| `compiler_trace_then_answer` | trace bootstrap followed by final-answer-only continuation |

Attention alignment is used only while trace loss is active. At evaluation time,
the compiler still reads the full hidden sequence and must attend for itself.

## Main Result

Run: `main_qwen35_attention_len4_retention`

Training: standard-template length-4 programs, 2048 bootstrap examples, 2048
answer-continuation examples, frozen Qwen3.5-4B features, independent step
queries, attention-aligned trace bootstrap.

| Variant | L=4 exec | L=8 exec | L=12 exec | L=24 exec | L=4 init | L=4 op | L=4 arg | L=4 exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `compiler_trace` | 99.6% | 0.4% | 1.2% | 0.8% | 100.0% | 100.0% | 99.9% | 99.6% |
| `compiler_answer_only` | 0.0% | 0.8% | 0.8% | 0.0% | 0.4% | 35.5% | 1.7% | 0.0% |
| `compiler_trace_then_answer` | 99.6% | 1.6% | 1.6% | 0.4% | 100.0% | 100.0% | 99.9% | 99.6% |

The direct answer control stays at chance:

| Split | Direct accuracy |
|---|---:|
| standard L=4 | 1.6% |
| standard L=8 | 1.2% |
| standard L=12 | 0.0% |
| standard L=24 | 1.2% |

## Stress Tests

The trained interface is narrow. When evaluated on paraphrased length-4 prompts,
the compiler does not transfer:

| Variant | paraphrase L=4 exec | paraphrase L=4 init | paraphrase L=4 op | paraphrase L=4 arg |
|---|---:|---:|---:|---:|
| `compiler_trace` | 1.6% | 94.9% | 58.0% | 54.0% |
| `compiler_trace_then_answer` | 2.0% | 94.5% | 62.9% | 55.7% |

Longer standard chains also fail when later step slots are not trained:

| Variant | standard L=8 exec | standard L=12 exec | standard L=24 exec |
|---|---:|---:|---:|
| `compiler_trace` | 0.4% | 1.2% | 0.8% |
| `compiler_trace_then_answer` | 1.6% | 1.6% | 0.4% |

A stronger length-12 trace condition improves per-step symbol accuracy but not
long-chain exactness:

| Split | Exec | Init | Op | Arg | Program exact |
|---|---:|---:|---:|---:|---:|
| standard L=4 | 99.2% | 100.0% | 100.0% | 99.8% | 99.2% |
| standard L=8 | 37.9% | 100.0% | 93.3% | 94.4% | 37.9% |
| standard L=12 | 5.9% | 100.0% | 86.3% | 88.9% | 4.3% |

This is a compounding-error regime: per-step symbol accuracy that looks high is
still not high enough for exact long-chain execution.

## Interpretation

The positive result is real but bounded. A small trainable module can compile
frozen Qwen hidden states into an executable latent program without span
features at inference time. Trace-time attention alignment is the critical
ingredient; without it, numeric extraction does not form reliably. Once the
interface is installed, final-answer-only continuation preserves it on the
trained distribution.

The negative result is equally important. The method does not yet produce a
general latent program interface. It does not discover the interface from
answer-only supervision, does not extrapolate to untrained step slots, and does
not handle paraphrased wording under the tested training budget.

## Conclusion

The experiment supports a narrow claim: frozen Qwen hidden states can feed a
span-free executable latent compiler when bootstrap supervision teaches both
symbols and attention. It does not support a broad claim of universal
posttraining improvement. The next technically meaningful step is to replace
independent step queries with a parser-like sequence tagger or to train a small
QLoRA adapter so Qwen exposes stable program-token features across wording and
length.

## Artifacts

- Source: `experiments/qwen_span_free_compiler/src/`
- Runs: `experiments/qwen_span_free_compiler/runs/`
- Analysis: `experiments/qwen_span_free_compiler/analysis/`
- Checkpoints: `large_artifacts/qwen_span_free_compiler/checkpoints/`
- Manifest: `experiments/qwen_span_free_compiler/checkpoint_manifest.csv`
