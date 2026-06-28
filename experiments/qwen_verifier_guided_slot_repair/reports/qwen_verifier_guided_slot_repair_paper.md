# Qwen Verifier-Guided Slot Repair

## Abstract

This experiment tests whether local verifier-guided search can recover exact long-chain execution from near-miss programs compiled by a QLoRA-adapted `Qwen/Qwen3-4B` model. Each prompt describes modular arithmetic updates. The compiler copies an initial value, operation sequence, and operation arguments from Qwen hidden states, then an invisible executor runs the copied program modulo 97.

The main result is positive as a headroom result. On fresh paired length-24 standard/paraphrase programs, the selected checkpoint scored 27.5% exact execution before repair and 91.0% after top-3/two-edit state-verifier repair. Repaired program exact was 90.0%, and repaired paired state consistency was 92.2%. A one-edit ablation reached 70.5%, showing that many failures are one local slot edit away and more are recoverable with two-slot edits.

This is not a deployable inference method yet. The primary verifier uses the true intermediate state trajectory. The result shows that the compiler's errors are often locally repairable, and that a learned or task-native verifier is a high-value next target.

## Question

If a Qwen-attached compiler is mostly right but brittle over long chains, how much exact execution can be recovered by searching a small neighborhood of locally plausible slot edits?

The tested repair space includes:

- alternate initial values from the compiler logits;
- alternate operations from the compiler logits;
- alternate arguments from the compiler logits;
- same-step operation and argument edits;
- pairs of argument edits.

The primary verifier requires the full intermediate state trajectory to match the true trajectory. This avoids a failure mode of final-answer-only verification: with many candidates and only 97 possible final answers, spurious final-answer matches are common.

## Method

The main run used `Qwen/Qwen3-4B` with 4-bit QLoRA adapters and a numeric-copy compiler. Training used a four-stage curriculum:

| Stage | Length range | Steps |
|---|---:|---:|
| short | 1-4 | 200 |
| medium | 1-8 | 200 |
| train | 1-12 | 200 |
| long | 8-24 | 300 |

The compiler was trained with trace supervision, executor loss, token-position selection losses, paired standard/paraphrase batches, and light intermediate-state supervision with weight 0.25. Checkpoints were saved at evaluation points and selected by validation `paired_len24_repair_executor_accuracy`.

Repair search used:

| Parameter | Value |
|---|---:|
| verifier | full state trajectory |
| candidate top-k per slot | 3 |
| maximum edits | 2 |
| max argument-pair slots | 24 |

The fresh retest used 256 standard length-24 programs, 256 paraphrase length-24 programs, and 256 paired length-24 latent programs rendered in both forms.

## Results

### Selected Validation Checkpoint

The selected checkpoint was step 800.

| Split | Unrepaired exact | Repaired exact | Repaired program exact | Repair found | Repair changed |
|---|---:|---:|---:|---:|---:|
| Standard L24 | 37.5% | 85.9% | 85.9% | 85.9% | 48.4% |
| Paraphrase L24 | 20.3% | 81.2% | 79.7% | 81.2% | 60.9% |
| Paired L24 | 30.5% | 88.3% | 88.3% | 88.3% | 57.8% |

Step 900 was worse on the primary repaired paired metric: 82.0%. Checkpoint selection therefore mattered.

### Fresh Retest

| Split | Unrepaired exact | Repaired exact | Repaired program exact | Repaired prefix | Repair found |
|---|---:|---:|---:|---:|---:|
| Fresh standard L24 | 27.3% | 88.3% | 87.9% | 90.7% | 88.3% |
| Fresh paraphrase L24 | 23.8% | 86.3% | 85.5% | 88.6% | 86.3% |
| Fresh paired L24 | 27.5% | 91.0% | 90.0% | 93.4% | 91.0% |

On the paired split, unrepaired paired state consistency was 69.1%; repaired paired state consistency was 92.2%. Repaired pair both-correct was 89.5%.

### One-Edit Ablation

The same selected checkpoint was retested with the same top-3 candidates but only one allowed edit.

| Split | Unrepaired exact | One-edit repaired exact | Two-edit repaired exact |
|---|---:|---:|---:|
| Fresh standard L24 | 27.3% | 64.1% | 88.3% |
| Fresh paraphrase L24 | 23.8% | 59.8% | 86.3% |
| Fresh paired L24 | 27.5% | 70.5% | 91.0% |

One-edit repair recovers a large fraction of failures, but two-edit repair is much stronger. This suggests the compiler often makes one or two local slot errors rather than globally incoherent programs.

## Interpretation

The repair result is much larger than the training-objective changes tested in this harness. The reason is straightforward: exact long-chain execution is an all-or-nothing metric, while the compiler's per-slot predictions are already close. At length 24, a small number of wrong arguments or operations can destroy the final answer. Local search converts the compiler's near-miss distribution into exact programs when the verifier can identify the correct state trajectory.

The repaired program-exact numbers are important. On fresh paired L24, repaired exact execution is 91.0% and repaired program exact is 90.0%. The search is usually recovering the true compiled program, not merely exploiting final-answer collisions.

The answer-only verifier pilot failed immediately and was stopped. With roughly 1,299 candidates and only 97 final answers, even weak candidates often include a spurious final-answer match. The state-trajectory verifier fixes that measurement problem by requiring the whole execution path to be correct.

## Limitations

The primary verifier is oracle-like. It uses the true intermediate state trajectory, which is available to the synthetic training and evaluation harness but would not be available for ordinary inference. Therefore this experiment should be read as a repair headroom test, not as a completed posttraining recipe.

The task is synthetic modular arithmetic. The compiler, token maps, operation set, and executor are all specialized. The result does not demonstrate broad intelligence improvement. It does show a concrete fact about this Qwen-attached runtime: most long-chain failures are locally repairable if a strong verifier is available.

## Next Step

The next experiment should replace the oracle state verifier with a learned verifier or repair policy. The most direct version is:

1. Generate candidate repair sets from the trained compiler.
2. Label candidates by whether their state trajectory is correct.
3. Train a verifier/reranker from Qwen hidden states, compiled slot logits, candidate edits, and execution features.
4. Evaluate whether the learned verifier can recover a meaningful fraction of the 91.0% oracle-repair ceiling without access to oracle states.

The success criterion should be fresh paired L24 exact execution substantially above the unrepaired 27.5% baseline, while preserving paired consistency.

## Artifacts

Small files live in:

```text
experiments/qwen_verifier_guided_slot_repair/
```

Large checkpoints live in:

```text
large_artifacts/qwen_verifier_guided_slot_repair/checkpoints/
```

Primary result files:

- `analysis/selected_checkpoints.csv`
- `analysis/selected_retest_metrics.csv`
- `analysis/selected_retest_metrics_one_edit.csv`
- `reports/qwen_verifier_guided_slot_repair_experiment_log.md`
