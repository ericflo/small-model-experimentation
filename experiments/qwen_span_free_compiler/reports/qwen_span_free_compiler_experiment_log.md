# Qwen Span-Free Compiler Experiment Log

## Objective

Test whether a frozen Qwen hidden sequence contains enough information for a
small trainable compiler to locate program-relevant tokens, emit executable
program symbols, and preserve that interface when trace loss is removed.

## Primary Questions

1. Can a learned query-attention compiler recover initial values, operations,
   and arguments from the full prompt hidden sequence without token-span input?
2. Does trace bootstrap install an executable interface that generalizes to
   longer chains?
3. Does answer-only continuation preserve or improve the installed interface?
4. Does answer-only training from scratch discover the interface?
5. How sensitive is the learned compiler to prompt wording and line-format
   changes?

## Metrics

- `executor_accuracy`: accuracy after argmax compilation and exact execution.
- `executor_target_mass`: differentiable executor probability assigned to the
  target answer.
- `init_accuracy`: compiled initial value accuracy.
- `op_accuracy`: per-step operation accuracy.
- `arg_accuracy`: per-step argument accuracy.
- `program_exact`: fraction of examples with all compiled symbols correct.
- `direct_accuracy`: direct answer classifier accuracy from the frozen Qwen
  answer-position feature.

## Artifact Layout

- Code and lightweight outputs:
  `experiments/qwen_span_free_compiler/`
- Checkpoints:
  `large_artifacts/qwen_span_free_compiler/checkpoints/`
- Run outputs:
  `experiments/qwen_span_free_compiler/runs/<run>/`
- Analysis outputs:
  `experiments/qwen_span_free_compiler/analysis/`

## 2026-06-21 Setup

Created the standalone experiment directory:

- `experiments/qwen_span_free_compiler/src/`
- `experiments/qwen_span_free_compiler/reports/`
- `experiments/qwen_span_free_compiler/runs/`
- `experiments/qwen_span_free_compiler/analysis/figures/`
- `large_artifacts/qwen_span_free_compiler/checkpoints/`

Implementation plan:

- Cache padded full-sequence hidden states and attention masks.
- Replace span-selected compiler inputs with learned query cross-attention over
  the whole sequence.
- Keep the direct answer head as a control on the answer-position feature.
- Add standard, mixed, and paraphrase prompt templates for wording-shift
  evaluation.
- Preserve staged variants: direct, trace, answer-only, trace-then-answer, and
  low-learning-rate trace-then-answer.

## 2026-06-21 Harness Smoke

Implemented the span-free harness:

- Full hidden-sequence feature caching with sequence masks.
- Learned query cross-attention compiler over the whole prompt.
- Direct answer classifier control from the answer-marker hidden state.
- Standard and paraphrase template evaluation splits.
- Staged variants for trace bootstrap and answer-only retention.
- External checkpoint writing.
- Analysis summary and figures.

Verification:

- Source compilation passed.
- Stale inherited-path scan passed.
- Tiny random Llama smoke completed.
- Analysis generation completed.
- Checkpoints were written under
  `large_artifacts/qwen_span_free_compiler/checkpoints/smoke_tiny/`.

Smoke interpretation:

- The smoke validates data flow, padded hidden-sequence caching, query-attention
  compiler calls, staged training, checkpointing, and analysis.
- The tiny random model has no meaningful parsing signal, so its low accuracy
  is only a plumbing check.

Next action: run a small Qwen pilot with frozen Qwen features to estimate
whether the span-free compiler can learn symbol extraction from the full prompt.

## 2026-06-21 Qwen Pilot: Query-Context Reader

Ran `pilot_qwen35_span_free` with frozen Qwen3.5-4B features, full-sequence
hidden-state caching, learned query-context attention, bootstrap training
lengths 1-3, answer-continuation training lengths 1-6, and evaluation lengths
3, 6, and 12 under standard and paraphrase templates.

Final standard-template metrics:

| Variant | L=3 exec | L=6 exec | L=12 exec | L=3 init | L=3 op | L=3 arg |
|---|---:|---:|---:|---:|---:|---:|
| `direct` | n/a | n/a | n/a | n/a | n/a | n/a |
| `compiler_trace` | 1.6% | 3.1% | 4.7% | 0.0% | 75.5% | 7.3% |
| `compiler_answer_only` | 3.1% | 0.0% | 0.0% | 0.0% | 29.2% | 0.0% |
| `compiler_trace_then_answer` | 0.0% | 0.0% | 1.6% | 1.6% | 81.8% | 0.5% |

Pilot interpretation:

- The direct answer control remains at chance.
- The trace-supervised compiler partially learns operation words, but numeric
  initial values and numeric arguments remain near chance on held-out examples.
- Training loss for initial values falls while held-out initial accuracy stays
  near chance, indicating memorization rather than a reusable numeric reader.
- The simplest query-context pooling is too weak for the span-free condition.

Next action: replace context-only pooling with token-logit pooling. Each learned
query still attends over the full prompt, but the class evidence is computed at
each token before pooling, which should preserve local numeric features.

## 2026-06-21 Qwen Pilot: Token-Logit Pooling

Patched the compiler so learned queries pool token-local class logits instead
of only pooling hidden-state contexts.

Ran `smoke_tiny_token_pool`; source compilation and tiny-model smoke passed.

Ran `pilot_qwen35_token_pool_trace` with the trace-supervised compiler only,
512 bootstrap examples, 512 continuation examples, standard-template training,
and standard/paraphrase evaluation.

Final metrics:

| Split | Exec | Target mass | Init | Op | Arg | Program exact |
|---|---:|---:|---:|---:|---:|---:|
| standard L=3 | 1.0% | 1.0% | 1.0% | 55.9% | 41.7% | 0.0% |
| standard L=6 | 0.0% | 1.0% | 3.1% | 56.4% | 35.4% | 0.0% |
| standard L=12 | 3.1% | 1.0% | 2.1% | 41.1% | 15.0% | 0.0% |
| paraphrase L=3 | 3.1% | 1.0% | 1.0% | 40.3% | 4.2% | 0.0% |
| paraphrase L=6 | 1.0% | 1.0% | 2.1% | 38.7% | 9.4% | 0.0% |
| paraphrase L=12 | 3.1% | 1.0% | 1.0% | 36.1% | 6.2% | 0.0% |

Pilot interpretation:

- Token-logit pooling helps argument extraction substantially compared with
  context-only pooling, but it does not solve initial-value extraction.
- The shared numeric token classifier is a likely bottleneck because initial
  values are uniform over the full modulus while arguments occupy a restricted
  range.

Next action: split initial-value and argument token classifiers, and expose
trace-loss weights so the hard initial-value channel can be emphasized.

## 2026-06-21 Alignment Bootstrap and Query Tests

Added optional attention-alignment supervision for trace-active stages. The
compiler still receives only the full hidden sequence at inference time; the
alignment labels are used only during bootstrap.

Key diagnostics:

| Run | Trained range | Main split | Exec | Init | Op | Arg | Program exact |
|---|---:|---|---:|---:|---:|---:|---:|
| `pilot_qwen35_attention_trace` | L=3 | standard L=3 | 99.2% | 100.0% | 100.0% | 99.7% | 99.2% |
| `pilot_qwen35_attention_trace` | L=3 | standard L=6 | 0.0% | 100.0% | 58.5% | 39.6% | 0.0% |
| `pilot_qwen35_generated_query_attention` | L=3 | standard L=3 | 99.2% | 100.0% | 100.0% | 99.7% | 99.2% |
| `pilot_qwen35_generated_query_attention` | L=3 | standard L=6 | 0.0% | 100.0% | 58.6% | 35.3% | 0.0% |

Interpretation:

- Attention-aligned bootstrap solves trained step slots.
- Generated step queries did not extrapolate to untrained later slots.
- Independent step queries are more appropriate for the capacity test.

Ran length-12 stress conditions:

| Run | Training template | Training range | Split | Exec | Init | Op | Arg | Program exact |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `pilot_qwen35_independent_attention_len12_mixed` | mixed | L=1-12 | standard L=12 | 1.6% | 100.0% | 57.7% | 47.5% | 0.0% |
| `pilot_qwen35_independent_attention_len12_fixed` | mixed | L=12 | standard L=12 | 0.0% | 100.0% | 60.1% | 51.2% | 0.0% |
| `pilot_qwen35_independent_attention_len12_standard` | standard | L=12 | standard L=12 | 3.1% | 100.0% | 80.5% | 82.5% | 0.8% |
| `pilot_qwen35_independent_attention_len12_standard_strong` | standard | L=12 | standard L=12 | 5.9% | 100.0% | 86.3% | 88.9% | 4.3% |

Interpretation:

- Initial-value extraction is solved by attention-aligned bootstrap.
- Longer chains fail mainly through compounding operation and argument errors.
- Mixed-template localization is materially harder than standard-template
  localization under the current budget.

Decision:

- Use standard-template, independent-query, attention-aligned bootstrap for the
  main retention run.
- Treat longer lengths and paraphrase wording as stress tests, not as solved
  conditions.

## 2026-06-21 Main Qwen Run

Ran `main_qwen35_attention_len4_retention` with frozen Qwen3.5-4B features,
full-sequence hidden-state caching, independent step queries, attention-aligned
trace bootstrap, and standard-template length-4 training.

Final standard-template metrics:

| Variant | L=4 exec | L=8 exec | L=12 exec | L=24 exec | L=4 init | L=4 op | L=4 arg | L=4 exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `compiler_trace` | 99.6% | 0.4% | 1.2% | 0.8% | 100.0% | 100.0% | 99.9% | 99.6% |
| `compiler_answer_only` | 0.0% | 0.8% | 0.8% | 0.0% | 0.4% | 35.5% | 1.7% | 0.0% |
| `compiler_trace_then_answer` | 99.6% | 1.6% | 1.6% | 0.4% | 100.0% | 100.0% | 99.9% | 99.6% |

Direct answer control:

| Split | Direct accuracy |
|---|---:|
| standard L=4 | 1.6% |
| standard L=8 | 1.2% |
| standard L=12 | 0.0% |
| standard L=24 | 1.2% |

Paraphrase stress result:

| Variant | paraphrase L=4 exec | paraphrase L=4 init | paraphrase L=4 op | paraphrase L=4 arg |
|---|---:|---:|---:|---:|
| `compiler_trace` | 1.6% | 94.9% | 58.0% | 54.0% |
| `compiler_trace_then_answer` | 2.0% | 94.5% | 62.9% | 55.7% |

Main interpretation:

- A span-free inference-time compiler can be installed when trace bootstrap
  includes attention alignment.
- Answer-only continuation preserves the installed length-4 standard-template
  interface.
- Answer-only training from scratch remains at chance.
- The learned interface is narrow: later step slots and paraphrased wording do
  not transfer under this configuration.

Final artifacts created:

- `analysis/final_metrics.csv`
- `analysis/summary.md`
- `analysis/figures/direct_accuracy.png`
- `analysis/figures/executor_accuracy.png`
- `analysis/figures/program_exact.png`
- `checkpoint_manifest.csv`

Next action: write the standalone report and run the final artifact audit.

## 2026-06-21 Final Audit

Final artifacts created:

- `reports/qwen_span_free_compiler_paper.md`
- `reports/qwen_span_free_compiler_paper.html`
- `checkpoint_manifest.csv`

Verification:

- Source compilation passed:
  `python -m py_compile src/qwen_span_free_compiler_experiment.py src/analyze_qwen_span_free_compiler.py`
- Checkpoint manifest validation passed for 23 saved checkpoints.
- No `.pt`, `.pth`, or `.ckpt` files are stored inside the lightweight
  experiment directory.
- Standalone wording scan passed on the report files, README, and source.
- Compile caches were removed after verification.

Artifact sizes:

- `experiments/qwen_span_free_compiler/`: 900K
- `large_artifacts/qwen_span_free_compiler/`: 274M

Conclusion:

The span-free inference-time compiler succeeds on the trained length-4 standard
template when trace bootstrap includes attention alignment. The main retention
row reaches 99.6% exact execution at length 4 after trace loss is removed,
matching the trace-supervised compiler. The direct answer head and answer-only
compiler remain at chance. The interface does not transfer to later step slots
or paraphrased prompts under the tested configuration.
