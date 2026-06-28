# Qwen Learned Repair Verifier

## Abstract

This experiment tests whether a small learned verifier can recover exact
long-chain execution from local repairs of programs compiled by a Qwen-attached
numeric compiler. Each prompt describes a sequence of modular arithmetic updates.
The compiler copies executable slots from the prompt, and a deterministic runtime
executes the copied program modulo 97.

The repair neighborhood contains 1,299 candidates for each length-24 example:
the base compiled program plus top-3 local edits of the initial value,
operations, arguments, same-step operation/argument pairs, and pairs of argument
edits. An oracle state verifier can identify the correct candidate when the full
target trajectory is available. This experiment trains a non-oracle verifier to
choose among the same candidates using only candidate features available at test
time.

The result is positive but partial. On fresh paired length-24 programs, base
execution accuracy is 30.3%. The learned verifier reaches 47.3%. A paired
consistency reranker reaches 51.0%. The oracle state-verifier ceiling is 88.1%.
Thus the learned verifier recovers 29.4% of the measured base-to-oracle gap on
the paired split.

## Question

Can a lightweight verifier learn to select useful local repairs of a compiled
program without seeing the true answer or the true state trajectory at test time?

This is the deployment-relevant version of verifier-guided repair. The oracle
state verifier is useful as a training label source and as a ceiling, but it is
not a runtime method. A learned verifier must infer which candidate is plausible
from model confidence geometry, edit type, candidate program structure, and
execution features.

## Method

The experiment freezes a QLoRA-attached `Qwen/Qwen3-4B` numeric compiler. The
compiler predicts:

- the initial modular value;
- one operation per step from add, subtract, multiply;
- one numeric argument per step.

The copied program is executed exactly modulo 97. All evaluated programs have
length 24.

For each example, the experiment enumerates a local repair neighborhood:

| Candidate class | Budget |
|---|---:|
| Base compiled program | 1 |
| Alternate initial values | top-3 |
| Alternate operations | top-3 per active step |
| Alternate arguments | top-3 per active step |
| Same-step operation and argument edits | top-3 by slot |
| Two-argument edits | top-3 by slot, up to 24 slots |

The resulting length-24 neighborhood has 1,299 candidates per example.

Offline labels mark candidates whose full execution trajectory matches the true
state trajectory. The learned verifier is a two-layer MLP trained with a groupwise
ranking loss:

```text
loss = logsumexp(all candidate scores) - logsumexp(correct candidate scores)
```

For examples with no correct candidate in the neighborhood, a small auxiliary
term selects the base program. At test time, labels are not used.

Candidate features include:

- candidate prior and prior delta from compiler logits;
- edit count, edit position, edit type, and changed-slot margins;
- changed values and argument-change magnitudes;
- candidate operation mix and argument statistics;
- differentiable executor support for the candidate's final answer and state
  path;
- agreement with the base program's answer and state path.

The primary main run uses 512 verifier-training examples, 128 validation
examples, and fresh held-out length-24 standard, paraphrase, and paired splits.

## Results

### Main Fresh Results

| Split | Base | Learned verifier | Pair rerank | Oracle ceiling |
|---|---:|---:|---:|---:|
| Fresh standard L24 | 28.5% | 44.1% | n/a | 90.6% |
| Fresh paraphrase L24 | 28.5% | 48.0% | n/a | 86.7% |
| Fresh paired L24 | 30.3% | 47.3% | 51.0% | 88.1% |

The learned verifier improves all fresh splits. On the paired split it recovers
29.4% of the base-to-oracle gap:

```text
(47.3 - 30.3) / (88.1 - 30.3) = 29.4%
```

The paired consistency reranker is only available when two renderings of the
same latent program are evaluated together. It selects from the learned
verifier's top candidates while rewarding program, state, and answer agreement
between renderings. On the paired split it improves executor accuracy from 47.3%
to 51.0%.

### Paired Split Details

| Metric | Base | Learned verifier | Pair rerank | Oracle ceiling |
|---|---:|---:|---:|---:|
| Executor accuracy | 30.3% | 47.3% | 51.0% | 88.1% |
| Program exact | 30.3% | 47.3% | 51.0% | 87.7% |
| State prefix fraction | 58.6% | 71.2% | 73.4% | 90.7% |
| Pair both-correct | 28.1% | 34.4% | 46.5% | 85.9% |
| Pair state consistency | 71.1% | 55.1% | 82.8% | 92.6% |

The individual learned verifier raises exact accuracy but lowers paired state
consistency, because it can choose different plausible repairs for two surface
forms. The paired reranker fixes much of that problem: paired state consistency
rises to 82.8%, and paired both-correct rises to 46.5%.

### Training Dynamics

The best validation epoch is epoch 15:

| Split | Base | Learned | Oracle |
|---|---:|---:|---:|
| Train L24 | 28.7% | 58.8% | 86.7% |
| Validation L24 | 32.0% | 50.8% | 85.2% |

The train/validation gap is visible but not dominant. Fresh held-out performance
lands between the validation result and the lower pilot results, which is
consistent with a real but incomplete learned signal.

## Iterations

The first smoke run used a tiny top-2/one-edit neighborhood to verify loading,
candidate generation, metric writing, and verifier checkpointing.

The first full-neighborhood pilot used fragment-mixed training and validation
renderings. That produced very low oracle ceilings on train and validation while
clean fresh standard/paraphrase ceilings stayed high. The setup was corrected to
use clean standard/paraphrase renderings for verifier training and validation.

A clean balanced pilot showed a measurable learned-verifier gain, but the gain
was weak on paraphrased prompts. A paraphrase-weighted pilot helped slightly but
reduced validation quality. The decisive improvement came from richer candidate
features that exposed edit values, operation mix, and argument-change magnitudes.
That configuration was used for the main run.

## Interpretation

The experiment answers the immediate question positively: a small learned
verifier can convert some oracle repair headroom into actual non-oracle accuracy.
The result is not just a validation artifact; it appears on fresh standard,
paraphrase, and paired length-24 splits.

The ceiling remains much higher than the learned result. The oracle verifier
finds correct repairs for 88.1% of fresh paired examples, while the learned
verifier reaches 47.3% and the paired reranker reaches 51.0%. This means the
candidate neighborhood is strong enough, but the learned selection signal is
still weak relative to exact trajectory verification.

The paired reranker result is important. When multiple renderings of the same
latent task are available, consistency is a strong non-oracle signal. It improves
both exact accuracy and state consistency. However, it is not a single-prompt
method.

## Limitations

- The task is synthetic modular arithmetic.
- The compiler and executor are specialized to this task family.
- The verifier labels are generated from true state trajectories offline.
- The learned verifier uses hand-engineered candidate features rather than a
  deeper model over prompt tokens and candidate traces.
- The main result is from one fixed compiler checkpoint and one main verifier
  seed.
- Paired consistency reranking requires two renderings of the same latent task.

## Conclusion

Learned repair verification is a high-value direction. It improves fresh paired
length-24 exact execution from 30.3% to 47.3%, and paired consistency reranking
raises it to 51.0%. This is a meaningful conversion of oracle repair headroom
into a non-oracle mechanism, but most of the 88.1% oracle ceiling remains
unclaimed.

The next step should make the verifier less hand-engineered and more
task-native: train a candidate-trace transformer or Qwen-conditioned reranker
over compact execution traces, and combine it with consistency training across
multiple prompt renderings.

## Artifacts

Small files live in:

```text
experiments/qwen_learned_repair_verifier/
```

Large checkpoints live in:

```text
large_artifacts/qwen_learned_repair_verifier/checkpoints/
```

Primary files:

- `analysis/summary.md`
- `analysis/final_metrics.csv`
- `analysis/all_final_metrics.csv`
- `analysis/figures/executor_accuracy.png`
- `runs/main_rich_learned_verifier_s512/metrics.csv`
- `runs/main_rich_learned_verifier_s512/verifier_train_log.csv`
- `checkpoint_manifest.csv`
