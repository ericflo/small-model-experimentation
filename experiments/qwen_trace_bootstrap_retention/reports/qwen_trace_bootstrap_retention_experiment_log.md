# Qwen Trace Bootstrap Retention Experiment Log

## Objective

Test whether a structured latent program interface can be installed with
symbol-trace supervision and then retained when training continues with only
final-answer supervision.

## Primary Questions

1. Does trace-supervised bootstrap train an executable latent compiler?
2. After trace loss is removed, does final-answer training preserve the
   compiled program interface?
3. Can answer-only continuation on longer chains improve length
   generalization without destroying symbol accuracy?
4. Does an answer-only compiler trained from scratch discover the same
   interface?
5. Which retained-interface failure dominates: initial value, operation,
   argument, or accumulated program exactness?

## Metrics

- `executor_accuracy`: accuracy after argmax compilation and exact latent
  execution.
- `executor_target_mass`: differentiable executor probability assigned to the
  target answer.
- `init_accuracy`: compiled initial value accuracy.
- `op_accuracy`: per-step operation accuracy.
- `arg_accuracy`: per-step argument accuracy.
- `program_exact`: fraction of examples with all compiled symbols correct.
- `direct_accuracy`: direct answer classifier accuracy from the same frozen
  Qwen features.

## Artifact Layout

- Code and lightweight outputs:
  `experiments/qwen_trace_bootstrap_retention/`
- Checkpoints:
  `large_artifacts/qwen_trace_bootstrap_retention/checkpoints/`
- Run outputs:
  `experiments/qwen_trace_bootstrap_retention/runs/<variant>/`
- Analysis outputs:
  `experiments/qwen_trace_bootstrap_retention/analysis/`

## 2026-06-21 Setup

Created the standalone experiment directory:

- `experiments/qwen_trace_bootstrap_retention/src/`
- `experiments/qwen_trace_bootstrap_retention/reports/`
- `experiments/qwen_trace_bootstrap_retention/runs/`
- `experiments/qwen_trace_bootstrap_retention/analysis/figures/`
- `large_artifacts/qwen_trace_bootstrap_retention/checkpoints/`

Next action: adapt the bridge harness for staged trace-bootstrap and
answer-only retention training.

## 2026-06-21 Harness Smoke

Implemented the staged harness:

- Separate bootstrap and answer-continuation training datasets.
- Matched variants:
  - `direct`: direct answer classifier on frozen Qwen features.
  - `compiler_answer_only`: latent compiler trained from final answer only.
  - `compiler_trace`: latent compiler with trace supervision throughout.
  - `compiler_trace_then_answer`: trace bootstrap followed by answer-only
    retention.
- Stage-aware training logs with `trace_loss_active`.
- External checkpoint writing.
- Analysis summary and figures.

Verification:

- Source compilation passed.
- Stale path scan over the new source and top-level docs passed.
- Tiny random Llama smoke completed.
- Analysis generation completed.
- Checkpoints were written under
  `large_artifacts/qwen_trace_bootstrap_retention/checkpoints/smoke_tiny/`.

Smoke interpretation:

- The smoke validates staged training mechanics, checkpointing, and analysis.
- The tiny random model has no meaningful parsing signal, so its accuracy is
  only a plumbing check.

Next action: run a small Qwen pilot to estimate whether answer-only
continuation preserves a trace-installed program interface.

## 2026-06-21 Qwen Pilot

Ran `pilot_qwen35_retention` with frozen Qwen3.5-4B features, bootstrap
training lengths 1-3, answer-continuation training lengths 1-6, and evaluation
lengths 3, 6, and 12.

| Variant | L=3 exec | L=6 exec | L=12 exec | L=12 init | L=12 op | L=12 arg | L=12 program exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| `direct` | 1.6% | 0.0% | 1.6% | n/a | n/a | n/a | n/a |
| `compiler_trace` | 100.0% | 98.4% | 92.2% | 93.8% | 99.7% | 100.0% | 92.2% |
| `compiler_answer_only` | 0.0% | 3.1% | 1.6% | 0.0% | 35.5% | 1.0% | 0.0% |
| `compiler_trace_then_answer` | 81.2% | 71.9% | 56.2% | 89.1% | 99.1% | 97.9% | 56.2% |

Pilot interpretation:

- Trace supervision installs a strong executable interface.
- Answer-only from scratch fails to discover the interface.
- Answer-only continuation retains a usable interface but degrades relative to
  keeping trace supervision, especially on length 12.
- In the retention variant, operation and argument parsing remain high; the
  largest degradation is program exactness through accumulated symbol errors.

Added `compiler_trace_then_answer_low_lr`, where the answer-only retention
stage uses a lower learning rate while sharing initialization and bootstrap
sampling with the normal retention variant.

Ran `pilot_qwen35_retention_lr` with the two retention variants:

| Variant | L=3 exec | L=6 exec | L=12 exec | L=12 init | L=12 op | L=12 arg | L=12 program exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| `compiler_trace_then_answer` | 76.6% | 78.1% | 56.2% | 89.1% | 99.1% | 97.7% | 56.2% |
| `compiler_trace_then_answer_low_lr` | 79.7% | 73.4% | 59.4% | 89.1% | 97.4% | 98.8% | 57.8% |

Pilot decision:

- Promote `compiler_trace_then_answer_low_lr` to the main run, while retaining
  the normal-LR row as a comparison.
- Use a larger bootstrap set and longer bootstrap to ensure the interface is
  well installed before removing trace loss.
- Evaluate out to length 24.

## 2026-06-21 Main Qwen Run

Ran `main_qwen35_retention` with frozen Qwen3.5-4B features, 1024 bootstrap
examples, 1024 answer-continuation examples, bootstrap training lengths 1-4,
answer-continuation training lengths 1-8, and evaluation lengths 4, 8, 12, and
24.

Final metrics:

| Variant | L=4 exec | L=8 exec | L=12 exec | L=24 exec | L=24 mass | L=24 init | L=24 op | L=24 arg | L=24 program exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct` | 0.8% | 0.8% | 1.2% | 0.4% | n/a | n/a | n/a | n/a | n/a |
| `compiler_trace` | 100.0% | 100.0% | 99.2% | 96.1% | 94.2% | 100.0% | 99.8% | 100.0% | 96.1% |
| `compiler_answer_only` | 0.8% | 1.2% | 1.2% | 0.8% | 1.0% | 0.8% | 33.7% | 0.0% | 0.0% |
| `compiler_trace_then_answer` | 100.0% | 100.0% | 99.6% | 96.9% | 95.3% | 100.0% | 99.9% | 100.0% | 96.9% |
| `compiler_trace_then_answer_low_lr` | 100.0% | 100.0% | 99.2% | 95.7% | 92.8% | 100.0% | 99.8% | 100.0% | 95.7% |

Length-24 stage trajectory:

| Variant | Stage | Step | L=24 exec | L=24 mass | L=24 op | L=24 arg |
|---|---|---:|---:|---:|---:|---:|
| `compiler_trace` | trace bootstrap | 800 | 85.2% | 82.4% | 99.3% | 100.0% |
| `compiler_trace` | trace continuation | 1600 | 96.1% | 94.2% | 99.8% | 100.0% |
| `compiler_trace_then_answer` | trace bootstrap | 800 | 87.1% | 83.5% | 99.4% | 100.0% |
| `compiler_trace_then_answer` | answer retention | 801 | 75.4% | 71.5% | 99.0% | 100.0% |
| `compiler_trace_then_answer` | answer retention | 1600 | 96.9% | 95.3% | 99.9% | 100.0% |
| `compiler_trace_then_answer_low_lr` | trace bootstrap | 800 | 87.1% | 83.5% | 99.4% | 100.0% |
| `compiler_trace_then_answer_low_lr` | answer retention | 801 | 89.8% | 85.5% | 99.6% | 100.0% |
| `compiler_trace_then_answer_low_lr` | answer retention | 1600 | 95.7% | 92.8% | 99.8% | 100.0% |

Main interpretation:

- Trace bootstrap installs an executable interface that generalizes to long
  chains before trace loss is removed.
- Answer-only continuation can preserve and improve the interface when it
  starts from the trace-installed state. The normal-LR retention row reaches
  96.9% exact execution at length 24.
- Answer-only training from scratch remains at chance, confirming that the
  interface is not discovered from sparse final-answer supervision alone under
  this setup.
- The low-LR retention row avoids the immediate post-removal drop, but the
  normal-LR row recovers and finishes slightly higher.
- Long-chain errors continue to track rare operation mistakes; initial value
  and argument parsing are exact at length 24.

Next action: write the standalone report, checkpoint manifest, and final audit.

## 2026-06-21 Final Audit

Final artifacts created:

- `reports/qwen_trace_bootstrap_retention_paper.md`
- `reports/qwen_trace_bootstrap_retention_paper.html`
- `checkpoint_manifest.csv`

Verification:

- Source compilation passed:
  `python -m py_compile src/qwen_trace_bootstrap_retention_experiment.py src/analyze_qwen_trace_bootstrap_retention.py`
- Checkpoint manifest validation passed for 15 saved checkpoints.
- Markdown and HTML report image references resolve.
- No `.pt`, `.pth`, or `.ckpt` files are stored inside the lightweight
  experiment directory.
- Standalone wording scan passed on the report files.
- Removed the compile cache after verification.

Artifact sizes:

- `experiments/qwen_trace_bootstrap_retention/`: 572K
- `large_artifacts/qwen_trace_bootstrap_retention/`: 130M

Conclusion:

The answer-only retention recipe succeeds after trace bootstrap. The main
`compiler_trace_then_answer` row reaches 96.9% exact execution at length 24,
while `compiler_answer_only` from scratch remains at 0.8% and the direct answer
head remains at 0.4%. This shows that final-answer supervision can preserve and
refine an installed latent program interface, but does not discover that
interface from scratch under this setup.
