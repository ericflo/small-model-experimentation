# Qwen Register-Token Structured Runtime

## Abstract

This experiment tests whether a Qwen-attached model can write an executable
program into fixed register tokens, then execute that program inside a fixed
cyclic modulo runtime. Each prompt describes a modular-arithmetic chain. The
prompt is followed by an appended register bank: one initial-value marker and
one operation and argument marker per possible step. A trainable compiler reads
only the hidden states at those marker positions. It predicts an initial
residue, operation routes, and arguments. A deterministic runtime executes the
predicted program modulo 97.

The main result is mixed. With QLoRA, full executable-slot supervision, full
intermediate-state supervision, and paired consistency between prompt renderings,
the model reaches 100.0% exact execution through length 12 on standard, paired,
and nearly all paraphrase splits. At length 24, it reaches 25.0% standard exact
execution, 5.5% paraphrase exact execution, and 11.7% paired exact execution.
A matched state-supervised control without paired consistency also solves length
12, but remains near chance at length 24: 3.9% standard, 1.6% paraphrase, and
1.6% paired.

The conclusion is narrow: paired trajectory consistency helps the fixed register
runtime start to generalize beyond the trained easy range, but it does not solve
long-chain prompt-invariant execution.

## Question

Can a Qwen-attached model configure a fixed structured runtime through invisible
register tokens, rather than relying on answer-token prediction?

The interface is intentionally strict. The compiler sees only:

- hidden state at `<REG_INIT>`;
- hidden state at each `<REG_OP_XX>`;
- hidden state at each `<REG_ARG_XX>`.

It does not receive hand-selected prompt spans. If the method works, the model
has learned to route source-program information into fixed latent registers.

## Task

Each example samples an initial value `x` modulo 97 and a chain of updates:

```text
x by a
-x by a
*x by a
```

The target answer is the final value after all active updates. Evaluation uses
lengths 4, 8, 12, and 24.

| Split | Meaning |
|---|---|
| Standard | Canonical prompt wording |
| Paraphrase | Alternative wording for the same operation semantics |
| Paired | Two renderings of each sampled program |

## Model

The base model is `Qwen/Qwen3-4B`, loaded in 4-bit NF4 with LoRA adapters. The
register compiler is a one-layer transformer over the fixed register bank:

- register width: 512 in the main run;
- attention heads: 4;
- max active steps: 24;
- outputs: init logits over 97 residues, operation logits over three primitives,
  and argument logits over 97 residues.

The runtime is fixed and cyclic. It applies predicted add, subtract, and
multiply primitives modulo 97. The runtime is differentiable during training,
so losses can be applied to the final answer distribution and every intermediate
state distribution.

## Training

The main run uses four losses:

| Loss | Purpose |
|---|---|
| Slot trace loss | Supervise init, operation, and argument registers |
| Final executor loss | Supervise the final runtime answer |
| State trajectory loss | Supervise every intermediate modulo state |
| Paired consistency loss | Align register and state distributions across two prompt renderings |

The main curriculum is:

| Stage | Lengths | Steps |
|---|---:|---:|
| Short | 1-4 | 150 |
| Medium | 1-8 | 150 |
| Train | 1-12 | 150 |
| Long | 8-24 | 150 |

The matched control uses the same setup but removes the paired consistency
losses.

## Results

### Main Run

| Split | Executor exact | Program exact | Init | Op | Arg | Prefix | Pair both | Pair state consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard L4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | n/a | n/a |
| Standard L8 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | n/a | n/a |
| Standard L12 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | n/a | n/a |
| Standard L24 | 25.0% | 25.0% | 100.0% | 93.8% | 89.7% | 80.5% | n/a | n/a |
| Paraphrase L4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | n/a | n/a |
| Paraphrase L8 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | n/a | n/a |
| Paraphrase L12 | 99.2% | 99.2% | 100.0% | 99.9% | 99.9% | 99.7% | n/a | n/a |
| Paraphrase L24 | 5.5% | 4.7% | 100.0% | 88.9% | 83.8% | 81.0% | n/a | n/a |
| Paired L4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Paired L8 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Paired L12 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Paired L24 | 11.7% | 10.2% | 100.0% | 89.7% | 85.7% | 79.1% | 1.6% | 1.6% |

The main run completely solves the trained length range. At length 24, the init
register is exact and operation/argument accuracy remain high, but exact program
execution is much lower. The model makes too many small slot errors for a
24-step chain.

### Matched Control

| Split | Main | No-pair control |
|---|---:|---:|
| Standard L24 executor exact | 25.0% | 3.9% |
| Standard L24 program exact | 25.0% | 2.3% |
| Paraphrase L24 executor exact | 5.5% | 1.6% |
| Paired L24 executor exact | 11.7% | 1.6% |
| Paired L24 pair both-correct | 1.6% | 0.0% |
| Paired L24 state consistency | 1.6% | 0.0% |

Both runs solve length 12. The difference appears at length 24: paired
consistency is load-bearing for the observed long-chain lift. It is not enough
to make the long-chain latent program stable.

## Training Dynamics

The main run developed in stages:

| Step | Stage | Standard L12 | Standard L24 | Paraphrase L24 | Paired L24 |
|---:|---|---:|---:|---:|---:|
| 150 | Short | 0.8% | 0.0% | 1.6% | 0.8% |
| 300 | Medium | 71.1% | 1.6% | 0.0% | 2.0% |
| 450 | Train | 63.3% | 0.0% | 0.8% | 0.0% |
| 451 | Long start | 93.0% | 13.3% | 4.7% | 6.6% |
| 600 | Long end | 100.0% | 25.0% | 5.5% | 11.7% |

The long stage is essential. The model does not extrapolate to length 24 from
short and medium training alone, even with state supervision.

## Interpretation

The experiment supports three claims.

First, the fixed register-token interface is trainable. The compiler reads only
the appended marker states and reaches perfect exact execution through length
12.

Second, state trajectory supervision alone is not enough for long-chain
generalization. The no-pair control has full state loss and still stays near
chance at length 24.

Third, paired consistency is useful but incomplete. It raises standard L24 from
3.9% to 25.0% and paired L24 from 1.6% to 11.7%, but paired both-correct and
state-consistency remain at 1.6%. The model can become more accurate without
becoming prompt-invariant.

The next technical target should be a repair or refinement mechanism over the
compiled register program. The current compiler often gets most of the length-24
slots right, but one or two wrong slots are enough to break exact execution.

## Limitations

- The task is synthetic modular arithmetic.
- The runtime is fixed and specialized.
- Training uses privileged intermediate state labels.
- The result is one main seed and one matched control seed.
- Length-24 paired consistency remains unsolved.

## Artifacts

Small files:

```text
experiments/qwen_register_structured_runtime/
```

Large checkpoints:

```text
large_artifacts/qwen_register_structured_runtime/checkpoints/
```

Primary outputs:

- `analysis/summary.md`
- `analysis/final_metrics.csv`
- `analysis/all_final_metrics.csv`
- `reports/qwen_register_structured_runtime_experiment_log.md`
- `reports/qwen_register_structured_runtime_paper.md`
- `checkpoint_manifest.csv`
