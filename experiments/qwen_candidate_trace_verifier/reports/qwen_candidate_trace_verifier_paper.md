# Qwen Candidate-Trace Verifier

## Abstract

This experiment tests whether a transformer verifier over candidate execution
traces can choose useful local repairs of programs compiled by a Qwen-attached
numeric compiler. Each prompt describes 24 modular arithmetic updates. A fixed
compiler copies an initial value, operation sequence, and argument sequence from
the prompt, and a deterministic runtime executes the copied program modulo 97.

For each prompt, the experiment enumerates 1,299 local repair candidates. Each
candidate contains a full executable program and a predicted 24-step state
trajectory. Offline labels identify candidates whose full state trajectory
matches the true trajectory. At test time, the verifier does not receive the true
answer or true states.

The result is positive. On fresh paired length-24 programs, the base compiler
reaches 30.3% exact execution. The candidate-trace verifier reaches 53.7%. A
paired consistency reranker reaches 56.2%. The oracle trajectory-verifier ceiling
is 88.1%. The trace verifier therefore recovers 40.5% of the base-to-oracle gap
on the paired split.

## Question

Can a learned verifier select the right local repair by reading the candidate
program and its execution trace, rather than relying on the true state trajectory
at test time?

This is a direct test of whether local repair headroom can be converted into a
non-oracle runtime mechanism. The verifier must decide among many plausible
nearby programs using only candidate-local signals.

## Setup

The fixed compiler reads a modular-arithmetic prompt and predicts:

- the initial value modulo 97;
- one operation per step from add, subtract, and multiply;
- one numeric argument per step.

The candidate search uses the compiler's top local alternatives:

| Candidate class | Budget |
|---|---:|
| Base compiled program | 1 |
| Alternate initial values | top-3 |
| Alternate operations | top-3 per active step |
| Alternate arguments | top-3 per active step |
| Same-step operation and argument edits | top-3 by slot |
| Two-argument edits | top-3 by slot, up to 24 slots |

For length 24, this produces 1,299 candidates per prompt.

## Verifier Architecture

Each candidate is represented as a 25-token trace:

- one global token containing candidate prior, edit count, initial value, final
  answer, and final soft-executor support;
- 24 step tokens containing candidate operation, argument, predicted state,
  base-program operation, base argument, base predicted state, edit flags,
  compiler log-probabilities, margins, entropies, and state soft support.

The model is a three-layer transformer encoder with width 128 and four attention
heads. It also receives aggregate candidate features through a small projection
layer. A groupwise ranking loss trains it to score correct trajectory candidates
above incorrect candidates:

```text
loss = logsumexp(all candidate scores) - logsumexp(correct candidate scores)
```

When no correct candidate exists in a group, a small auxiliary term selects the
base program.

## Results

### Main Fresh Results

| Split | Base | Trace verifier | Pair rerank | Oracle ceiling |
|---|---:|---:|---:|---:|
| Fresh standard L24 | 28.5% | 50.4% | n/a | 90.6% |
| Fresh paraphrase L24 | 28.5% | 55.5% | n/a | 86.7% |
| Fresh paired L24 | 30.3% | 53.7% | 56.2% | 88.1% |

The trace verifier improves every fresh split. On the paired split, it recovers:

```text
(53.7 - 30.3) / (88.1 - 30.3) = 40.5%
```

of the measured base-to-oracle gap.

### Fresh Paired Details

| Metric | Base | Trace verifier | Pair rerank | Oracle ceiling |
|---|---:|---:|---:|---:|
| Executor accuracy | 30.3% | 53.7% | 56.2% | 88.1% |
| Program exact | 30.3% | 53.7% | 56.2% | 87.7% |
| State prefix fraction | 58.6% | 76.4% | 78.0% | 90.7% |
| Pair both-correct | 28.1% | 39.8% | 52.3% | 85.9% |
| Pair state consistency | 71.1% | 58.2% | 87.1% | 92.6% |

The single-prompt trace verifier improves exact execution but reduces paired
state consistency: two prompt renderings may choose different local repairs. A
paired reranker that rewards agreement among the top trace-verifier candidates
improves executor accuracy to 56.2%, paired both-correct accuracy to 52.3%, and
paired state consistency to 87.1%.

### Training Dynamics

The main run used 512 training examples and 128 validation examples. The best
validation epoch was epoch 15.

| Split | Base | Trace verifier | Oracle |
|---|---:|---:|---:|
| Train L24 | 28.7% | 58.6% | 86.7% |
| Validation L24 | 32.0% | 56.2% | 85.2% |

The fresh held-out paired result, 53.7%, is close to validation, suggesting that
the learned signal transfers to new length-24 programs.

## Iterations

The smoke run used a tiny top-2/one-edit candidate space to validate loading,
trace construction, training, checkpointing, and metric writing.

The first full-neighborhood pilot used a two-layer trace verifier with 128
training examples. It reached 53.1% validation accuracy but only 40.6% fresh
paired accuracy.

The second pilot used a deeper three-layer trace verifier and 192 training
examples. It reached 54.2% validation accuracy and 44.5% fresh paired accuracy.
That configuration was selected for the main run.

The main run scaled the selected configuration to 512 training examples and
larger fresh held-out splits. It reached 53.7% fresh paired accuracy and 56.2%
with paired reranking.

## Interpretation

The experiment supports the central hypothesis: candidate execution traces
contain useful non-oracle information for repair selection. A small transformer
can read those traces and choose better repairs than the base compiler's top
program.

The result also shows that selection remains the bottleneck. The oracle ceiling
is 88.1% on fresh paired programs, while the trace verifier reaches 53.7%. The
candidate neighborhood contains correct programs far more often than the learned
verifier can identify them.

Pair consistency is a powerful additional signal when multiple renderings of the
same latent program are available. It substantially improves paired both-correct
accuracy and state consistency. It is not, however, a single-prompt method.

## Limitations

- The task is synthetic modular arithmetic.
- The compiler and executor are specialized.
- Offline labels use true state trajectories.
- The verifier still uses engineered trace fields, not raw prompt tokens.
- The main result is one fixed compiler checkpoint and one main verifier seed.
- Paired reranking requires two renderings of the same latent program.

## Conclusion

A candidate-trace verifier improves fresh paired length-24 exact execution from
30.3% to 53.7%, and paired consistency reranking raises it to 56.2%. This is a
substantial conversion of local repair headroom into a non-oracle mechanism, but
the 88.1% oracle ceiling shows that much better selection is still possible.

The next step should train the verifier and compiler together: either distill the
trace verifier's successful choices back into the compiler, or train a verifier
that consumes compact prompt-conditioned representations in addition to the
candidate trace.

## Artifacts

Small files live in:

```text
experiments/qwen_candidate_trace_verifier/
```

Large checkpoints live in:

```text
large_artifacts/qwen_candidate_trace_verifier/checkpoints/
```

Primary files:

- `analysis/summary.md`
- `analysis/final_metrics.csv`
- `analysis/all_final_metrics.csv`
- `analysis/figures/executor_accuracy.png`
- `runs/main_trace_verifier_s512/metrics.csv`
- `runs/main_trace_verifier_s512/verifier_train_log.csv`
- `checkpoint_manifest.csv`
