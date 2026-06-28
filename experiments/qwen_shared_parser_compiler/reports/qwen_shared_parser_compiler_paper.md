# Shared Parser Compiler for Frozen Qwen Hidden States

## Abstract

This experiment tests whether a frozen Qwen3.5-4B model can support a small
trainable latent compiler that reads the full hidden sequence, recovers an
ordered modular arithmetic program, and executes that program without emitting
intermediate text. The best trace-supervised parser reaches 78.5% exact
execution at length 4, 62.5% at length 8, and 39.1% at length 12 on the
standard template. It falls to 0.4% at length 24 and does not transfer to
paraphrased templates. Direct answer classification and answer-only parser
training stay near 97-way chance. Answer-only continuation from a
trace-installed parser is destructive for longer lengths.

## Question

The experiment asks whether frozen Qwen hidden states contain enough structure
for a small posttraining head to compile text into an executable latent program
without being given token spans at inference time.

The target behavior is not just answer classification. The parser must predict:

- The initial residue modulo 97.
- One operation symbol per step: add, subtract, or multiply.
- One argument residue per step.
- Ordered slots for operation and argument tokens.

An exact modular executor then applies the compiled program.

## Task

Each prompt describes a hidden value `x` modulo 97. A prompt gives an initial
value and a list of update steps. The answer is the final value after all
updates.

Training prompts use the standard wording:

- `Initial x = n.`
- `Step: add k.`
- `Step: subtract k.`
- `Step: multiply by k.`

Evaluation uses both this standard wording and paraphrased wording. The main
training lengths are 1 through 12. Evaluation lengths are 4, 8, 12, and 24.

## Model

Qwen3.5-4B is loaded frozen in 4-bit mode. The trainable components are small
heads over cached hidden states.

The final parser uses:

- A shared token MLP over every hidden token.
- Token role heads for init, operation, and argument positions.
- Token symbol heads for initial values, operation classes, and argument values.
- A monotonic slot reader for operation slots.
- An after-operation argument reader that anchors each argument slot to the
  learned operation slot and scores nearby following tokens.
- An exact differentiable modular executor for answer loss.

The parser receives full hidden sequences and attention masks. It does not
receive gold token spans at inference time.

## Training

The main trace run uses:

- Model: `Qwen/Qwen3.5-4B`
- Modulus: 97
- Max steps: 24
- Trace train lengths: 1-12
- Trace train examples: 4096
- Eval examples per split: 256
- Parser width: 768
- Trace steps: 1600
- Argument trace loss weight: 4
- Argument selection loss weight: 4
- Argument reader: after-operation window, width 8

Controls:

- `direct`: answer classifier from the frozen answer-marker hidden state.
- `compiler_answer_only`: parser and executor trained only from final answer.
- `compiler_trace`: parser trained with symbol and selection trace labels.
- `compiler_trace_then_answer`: trace bootstrap followed by answer-only training.
- `compiler_trace_then_answer_low_lr`: same, with 0.1x answer-stage learning rate.

## Main Results

Standard-template exact answer accuracy:

| Variant | L=4 | L=8 | L=12 | L=24 |
|---|---:|---:|---:|---:|
| Direct answer head | 2.0% | 0.0% | 1.2% | 1.6% |
| Trace parser | 78.5% | 62.5% | 39.1% | 0.4% |
| Answer-only parser | 2.3% | 2.3% | 2.0% | 3.1% |

Paraphrase-template exact answer accuracy:

| Variant | L=4 | L=8 | L=12 | L=24 |
|---|---:|---:|---:|---:|
| Direct answer head | 1.2% | 0.8% | 2.0% | 1.2% |
| Trace parser | 4.3% | 0.8% | 2.3% | 0.4% |
| Answer-only parser | 1.2% | 1.6% | 1.2% | 1.6% |

The trace parser learns a real executable interface on the trained wording and
length range. The direct and answer-only controls do not.

## Parser Diagnostics

Trace parser diagnostics on standard-template splits:

| Split | Init | Init pos | Op | Op pos | Arg | Arg pos | Program exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| L=4 | 100.0% | 100.0% | 99.9% | 100.0% | 94.0% | 95.1% | 78.1% |
| L=8 | 100.0% | 100.0% | 99.2% | 100.0% | 94.8% | 94.8% | 62.1% |
| L=12 | 100.0% | 100.0% | 99.0% | 100.0% | 93.2% | 93.7% | 38.7% |
| L=24 | 100.0% | 100.0% | 82.6% | 72.9% | 70.7% | 68.8% | 0.0% |

The parser solves initial values and operation slots inside the trained length
range. Argument errors are small per step but compound strongly. At length 24,
operation and argument localization also degrade.

Paraphrase diagnostics show a different failure. Init token selection remains
high, but operation and argument slot selection break under wording shift. This
indicates that the parser is format-sensitive rather than template-invariant.

## Retention

Answer-only continuation after trace bootstrap:

| Variant | Split | L=4 | L=8 | L=12 | L=24 |
|---|---|---:|---:|---:|---:|
| Trace then answer | standard | 60.2% | 0.4% | 1.2% | 0.8% |
| Trace then answer, low LR | standard | 78.1% | 6.6% | 0.8% | 0.0% |
| Trace then answer | paraphrase | 3.1% | 2.0% | 2.0% | 1.2% |
| Trace then answer, low LR | paraphrase | 2.0% | 0.8% | 0.4% | 1.2% |

Answer-only continuation does not reliably preserve the parser. The low
learning-rate variant preserves length 4 but loses longer standard lengths by
the end of continuation.

## Interpretation

The strongest positive result is that a small trace-supervised parser can read
frozen Qwen hidden states and drive exact latent execution without span inputs
at inference time. This is a genuine latent compiler result for the standard
template through length 12.

The strongest negative result is that the learned parser is not robust. It does
not extrapolate to length 24, does not transfer to paraphrased wording, and is
not discovered from answer-only training. Final-answer continuation also
damages the installed interface.

The main bottleneck is not modular execution. It is reliable parsing. Exact
execution falls quickly when per-step operation and argument accuracies are
below the high 90s, because every compiled symbol must be correct.

## Conclusion

Frozen Qwen3.5-4B hidden states support a trace-supervised latent program
compiler, but this parser is still too brittle to count as a broad intelligence
gain recipe. The result supports the narrower claim that structured trace
supervision can install an executable latent interface. It does not support the
stronger claim that final-answer posttraining alone discovers or preserves such
an interface.

The next best test is to train the parser inside Qwen with a small QLoRA
adapter, rather than only training external heads over frozen hidden states.
