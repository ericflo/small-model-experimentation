# Qwen Structured Bridge Experiment Log

## Objective

Test whether a frozen Qwen encoder can be attached to a trainable structured
latent executor. The bridge reads hidden states from the prompt, predicts a
modular initial value and per-step program symbols, and the executor computes
the answer without emitting intermediate text.

## Primary Questions

1. Can a small bridge compile Qwen hidden states into executable modular
   program symbols?
2. Does structured execution generalize to longer operation chains better than
   a direct answer classifier trained on the same frozen Qwen features?
3. Is trace supervision necessary, or can answer-only supervision discover the
   latent program interface?
4. Which failure mode dominates: initial-value parsing, operation parsing,
   argument parsing, or accumulated execution error?

## Metrics

- `direct_accuracy`: direct answer classifier accuracy from Qwen features.
- `executor_accuracy`: accuracy after argmax compilation and exact latent
  execution.
- `executor_target_mass`: soft executor probability assigned to the target
  answer.
- `init_accuracy`: compiled initial value accuracy.
- `op_accuracy`: per-step operation accuracy.
- `arg_accuracy`: per-step argument accuracy.
- `program_exact`: fraction of examples with all compiled symbols correct.

## Artifact Layout

- Code and lightweight outputs:
  `experiments/qwen_structured_bridge/`
- Checkpoints:
  `large_artifacts/qwen_structured_bridge/checkpoints/`
- Run outputs:
  `experiments/qwen_structured_bridge/runs/<variant>/`
- Analysis outputs:
  `experiments/qwen_structured_bridge/analysis/`

## 2026-06-21 Setup

Created the standalone experiment directory:

- `experiments/qwen_structured_bridge/src/`
- `experiments/qwen_structured_bridge/reports/`
- `experiments/qwen_structured_bridge/runs/`
- `experiments/qwen_structured_bridge/analysis/figures/`
- `large_artifacts/qwen_structured_bridge/checkpoints/`

Installed `peft` so LoRA can be added as a later condition if frozen-feature
compilation bottlenecks on parsing. The first implementation keeps Qwen frozen
to isolate whether the structured bridge works before training Qwen weights.

Next action: implement the Qwen-to-executor bridge harness and run a tiny
smoke test.

## 2026-06-21 Harness Smoke

Implemented the bridge harness:

- Text modular-program generator with line-boundary token positions.
- Frozen-model hidden-state extraction for init, step, and answer lines.
- Direct answer classifier control.
- Program compiler heads for initial value, operation, and argument symbols.
- Differentiable soft modular executor and argmax exact executor.
- Variants: `direct`, `compiler_trace`, and `compiler_answer_only`.
- External checkpoint writing.
- Analysis summary and figures.

Verification:

- Source compilation passed.
- `peft` import passed, version `0.19.1`.
- Tiny random Llama smoke completed with all three variants.
- Analysis generation completed.
- Checkpoints were written under
  `large_artifacts/qwen_structured_bridge/checkpoints/smoke_tiny/`.

Smoke interpretation:

- The smoke validates data flow, hidden-state extraction, training,
  checkpointing, and analysis.
- The tiny random model has no useful parsing signal, so its low accuracy is
  not an experimental result.

Next action: run a small Qwen pilot to test whether frozen Qwen hidden states
support executable program compilation.

## 2026-06-21 Qwen Pilot

Ran `pilot_qwen35_frozen_bridge` with frozen Qwen3.5-4B features, training
lengths 1-3, and evaluation lengths 3, 6, and 8.

Initial line-boundary feature result:

| Variant | L=3 direct | L=3 executor | L=6 executor | L=8 executor | Init acc | Op acc | Arg acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `direct` | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a |
| `compiler_trace` | n/a | 2.1% | 0.0% | 0.0% | 0.0-4.2% | 95.6-99.7% | 8.7-14.6% |
| `compiler_answer_only` | n/a | 2.1% | 2.1% | 0.0% | 0.0-4.2% | 32.3-33.3% | 0.0-0.5% |

Interpretation:

- Qwen line-end features made operation words easy to classify.
- Numeric initial values and numeric arguments were not recoverable from that
  feature choice under the small pilot budget.
- The executor path itself was not the failure; it was being configured with
  wrong numeric symbols.

Patched the harness to read hidden states at numeric token spans for the
initial value and step arguments, while keeping operation prediction on the
operation-line prefix.

Ran `pilot_qwen35_numeric_spans` with the patched feature extractor.

Numeric-span result:

| Variant | L=3 direct | L=3 executor | L=6 executor | L=8 executor | Init acc | Op acc | Arg acc | Program exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct` | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `compiler_trace` | n/a | 71.9% | 76.6% | 71.9% | 71.9-85.9% | 98.8-100.0% | 99.5-100.0% | 71.9-76.6% |
| `compiler_answer_only` | n/a | 0.0% | 4.7% | 0.0% | 0.0-3.1% | 33.4-40.1% | 0.0-0.2% | 0.0% |

Pilot interpretation:

- Frozen Qwen hidden states can support executable program compilation when
  the bridge reads the numeric-token features directly.
- Trace supervision is doing the important work. Answer-only supervision did
  not discover the latent program interface.
- The main remaining bottleneck is initial-value classification over 97
  residues. Operation and argument parsing are already near exact.

Main decision:

- Run a larger Qwen numeric-span bridge with more examples per residue.
- Train on lengths 1-4 and evaluate lengths 4, 8, and 12.
- Keep `direct`, `compiler_trace`, and `compiler_answer_only` to preserve the
  direct-answer and answer-only controls.

## 2026-06-21 Main Qwen Run

Ran `main_qwen35_numeric_spans` with frozen Qwen3.5-4B features, 1024 training
examples, training lengths 1-4, and evaluation lengths 4, 8, and 12.

| Variant | L=4 accuracy | L=8 accuracy | L=12 accuracy | L=12 target mass | L=12 init | L=12 op | L=12 arg | L=12 program exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct` | 0.0% | 1.2% | 0.4% | n/a | n/a | n/a | n/a | n/a |
| `compiler_trace` | 100.0% | 99.2% | 95.7% | 94.1% | 100.0% | 99.6% | 100.0% | 95.7% |
| `compiler_answer_only` | 1.6% | 0.0% | 0.8% | 1.0% | 1.2% | 33.5% | 3.1% | 0.0% |

Main interpretation:

- The trace-supervised compiler/executor is the first strong Qwen-attached
  result in this line: the frozen model supplies hidden features that a small
  bridge can compile into an executable latent program.
- The direct answer head remains at 97-way chance from the same frozen Qwen
  features.
- Answer-only latent compilation also remains at chance. Final-answer reward
  alone did not discover the discrete program interface under this setup.
- At length 12, the residual error is almost entirely rare operation
  misclassification; initial value and numeric argument parsing are exact.

## 2026-06-21 Length Scale Check

Ran `scale_qwen35_length24_trace` with the trace-supervised compiler only,
training lengths 1-4, and evaluation lengths 4, 12, 16, and 24.

| L | Executor accuracy | Target mass | Init acc | Op acc | Arg acc | Program exact |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 100.0% | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% |
| 12 | 93.8% | 93.6% | 100.0% | 99.5% | 100.0% | 93.8% |
| 16 | 92.2% | 91.5% | 100.0% | 99.6% | 100.0% | 92.2% |
| 24 | 87.5% | 82.7% | 100.0% | 99.5% | 100.0% | 87.5% |

Scale interpretation:

- The bridge generalizes well beyond the training length range, reaching 87.5%
  exact execution at 24 steps after training only on 1-4 step programs.
- The remaining error scales like accumulated rare operation mistakes.
- Numeric parsing is not the bottleneck in the scaled run: initial value and
  argument accuracy are 100.0% at length 24.

Next action: write the standalone report, checkpoint manifest, and final audit.

## 2026-06-21 Final Audit

Final artifacts created:

- `reports/qwen_structured_bridge_paper.md`
- `reports/qwen_structured_bridge_paper.html`
- `checkpoint_manifest.csv`

Verification:

- Source compilation passed:
  `python -m py_compile src/qwen_structured_bridge_experiment.py src/analyze_qwen_structured_bridge.py`
- Checkpoint manifest validation passed for 16 saved checkpoints.
- Markdown and HTML report image references resolve.
- No `.pt`, `.pth`, or `.ckpt` files are stored inside the lightweight
  experiment directory.
- Standalone wording scan passed on the report files.
- Removed the compile cache after verification.

Artifact sizes:

- `experiments/qwen_structured_bridge/`: 452K
- `large_artifacts/qwen_structured_bridge/`: 98M

Conclusion:

Frozen Qwen3.5-4B hidden states can drive a small trace-supervised structured
bridge that compiles text into an executable latent modular program. The main
run reaches 95.7% exact execution at length 12 after training on lengths 1-4.
The length-24 scale check reaches 87.5%. Direct answer classification and
answer-only latent compilation remain at chance.
