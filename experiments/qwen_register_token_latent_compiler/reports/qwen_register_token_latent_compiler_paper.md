# Qwen Register-Token Latent Compiler

## Abstract

This experiment tests whether a small Qwen-attached training intervention can
make a fixed bank of latent register tokens act like an executable program
surface. Each prompt describes a modular-arithmetic chain. The prompt is
followed by marker tokens for an initial-value register and one operation and
argument register per possible step. A trainable compiler reads only the hidden
states at those register markers. It predicts a program, and a deterministic
invisible executor runs that program modulo 97.

The result is mixed. With QLoRA adaptation and trace supervision, the model
learns the register interface well through length 12: exact execution reaches
88.3-94.5% on standard prompts, 89.1-90.6% on paraphrased prompts, and
87.9-96.1% on paired renderings of the same program. At length 24, exact
execution falls to 21.9% on standard prompts, 3.1% on paraphrases, and 12.5% on
paired prompts. Per-slot accuracy remains high at length 24, but exact
long-chain program correctness is brittle.

Two controls stayed near chance. A direct answer head trained on the answer
marker reached only 3.1% on standard length 24. The same register compiler
trained only from final-answer loss reached 1.6% on standard length 24 and 0.0%
on paired length 24. The useful signal came from supervised executable traces,
not from final-answer supervision alone.

## Question

Can a Qwen-attached model learn to write an invisible executable program into a
fixed register bank, where the downstream runtime reads only those register
hidden states and never reads hand-selected prompt spans?

This is a strict interface test. The bridge sees:

- the hidden state at `<REG_INIT>`;
- the hidden state at each `<REG_OP_XX>`;
- the hidden state at each `<REG_ARG_XX>`.

It does not receive token spans for the source numbers or operation words. If it
works, the language model has learned to route source information into fixed
latent program slots.

## Task

Each example samples an initial value `x` modulo 97 and a chain of updates:

```text
+x by a
-x by a
*x by a
```

The true answer is the final value after executing all active steps modulo 97.
The maximum register bank has 24 steps. Evaluation uses lengths 4, 8, 12, and
24, with three rendering modes:

| Split | Meaning |
|---|---|
| Standard | Canonical prompt wording |
| Paraphrase | Alternative wording for the same operation semantics |
| Paired | Two renderings of each sampled program, used to measure consistency |

## Model

The base model is `Qwen/Qwen3-4B`, loaded in 4-bit NF4 with LoRA adapters on
linear modules. The main compiler is a one-layer transformer over the register
bank:

- input: Qwen hidden states at register-marker positions;
- width: 512;
- heads: 4;
- outputs: one init distribution over 97 values, one operation distribution per
  step, and one argument distribution per step.

A deterministic differentiable executor maps the predicted distributions into a
final answer distribution. Argmax execution is used for exact program metrics.

## Training Variants

| Variant | Interface | Supervision |
|---|---|---|
| Direct answer head | Answer-marker hidden state | Final answer only |
| Register answer-only | Register hidden states | Final answer through soft executor |
| Register trace | Register hidden states | Init, operation, argument, and final answer |

The main run uses the register-trace variant. It trains for 600 optimizer steps
with a curriculum: 150 steps on lengths 1-4, 150 on 1-8, 150 on 1-12, and 150 on
8-24.

## Main Results

### Trace-Supervised Register Compiler

| Split | Executor exact | Program exact | Init | Op | Arg | Prefix | Pair both | Pair state consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard L4 | 88.3% | 88.3% | 88.3% | 100.0% | 100.0% | 88.3% | n/a | n/a |
| Standard L8 | 94.5% | 94.5% | 94.5% | 100.0% | 100.0% | 94.5% | n/a | n/a |
| Standard L12 | 94.5% | 94.5% | 94.5% | 100.0% | 100.0% | 94.5% | n/a | n/a |
| Standard L24 | 21.9% | 21.1% | 91.4% | 97.3% | 95.2% | 83.9% | n/a | n/a |
| Paraphrase L4 | 90.6% | 90.6% | 90.6% | 100.0% | 100.0% | 90.6% | n/a | n/a |
| Paraphrase L8 | 91.4% | 91.4% | 91.4% | 100.0% | 100.0% | 91.4% | n/a | n/a |
| Paraphrase L12 | 89.1% | 89.1% | 89.1% | 100.0% | 100.0% | 89.1% | n/a | n/a |
| Paraphrase L24 | 3.1% | 2.3% | 89.1% | 92.5% | 88.5% | 72.9% | n/a | n/a |
| Paired L4 | 87.9% | 87.9% | 87.9% | 100.0% | 100.0% | 87.9% | 87.5% | 99.2% |
| Paired L8 | 96.1% | 96.1% | 96.1% | 100.0% | 100.0% | 96.1% | 96.1% | 99.2% |
| Paired L12 | 91.0% | 91.0% | 91.0% | 100.0% | 100.0% | 91.0% | 90.6% | 98.4% |
| Paired L24 | 12.5% | 12.1% | 92.2% | 94.7% | 92.4% | 79.5% | 1.6% | 3.1% |

The trace-supervised compiler clearly learns the latent register interface
through length 12. It also learns many length-24 slots: init, op, and arg
accuracies are all high on the standard length-24 split. But exact length-24
program execution is much lower, because a long chain needs every crucial slot
and every intermediate transition to be right.

### Controls

| Control | Standard L4 | Standard L8 | Standard L12 | Standard L24 | Paraphrase L24 | Paired L24 |
|---|---:|---:|---:|---:|---:|---:|
| Direct answer head | 1.6% | 1.6% | 1.6% | 3.1% | 0.0% | 1.6% |
| Register answer-only | 0.0% | 0.0% | 0.0% | 1.6% | 0.0% | 0.0% |

Modulo-97 chance is about 1.0%. These controls do not learn the task under the
tested budgets. The answer-only register compiler does learn a stable but
uninformative program pattern: paired consistency is high because both prompt
renderings collapse to the same wrong registers.

## Training Dynamics

The main run becomes useful only after enough curriculum exposure:

| Step | Standard L4 | Standard L8 | Standard L12 | Standard L24 | Paired L24 |
|---|---:|---:|---:|---:|---:|
| 450 | 59.4% | 56.2% | 64.1% | 2.3% | 1.6% |
| 600 | 88.3% | 94.5% | 94.5% | 21.9% | 12.5% |

The last 150-step long-chain stage is responsible for the length-24 lift, but it
does not produce prompt-invariant length-24 programs.

## Interpretation

This is evidence that a local 4B-scale model can be trained to expose a
program-like latent interface through fixed register tokens. The strongest
result is not the length-24 score; it is the length-12 behavior combined with
the failed controls. The bridge reads only register-marker states, yet it
recovers complete executable programs for unseen examples and paraphrases.

The hard failure is compositional reliability. At length 24, per-slot accuracies
near 90-97% are not enough. The product of many small slot errors destroys exact
execution, and paired consistency shows that different prompt renderings do not
land on the same latent state trajectory.

This points to a specific next direction: the register interface should be kept,
but training should directly penalize long-chain trajectory errors and paired
state disagreement. More final-answer-only optimization is unlikely to discover
the interface by itself; the answer-only control is the clearest evidence for
that.

## Limitations

- The task is synthetic modular arithmetic, not open-domain language use.
- Trace supervision supplies privileged intermediate labels during training.
- The executor is fixed and exact; the experiment does not learn a general
  runtime.
- The main run is a single 600-step configuration, not a scaling study.
- Length-24 performance is not robust enough to call the method solved.

## Artifacts

- Source, metadata, analysis, and reports:
  `experiments/qwen_register_token_latent_compiler/`
- Checkpoints:
  `large_artifacts/qwen_register_token_latent_compiler/checkpoints/`
- Aggregate metrics:
  `experiments/qwen_register_token_latent_compiler/analysis/all_final_metrics.csv`
- Main summary:
  `experiments/qwen_register_token_latent_compiler/analysis/summary.md`
