# Qwen Shared Parser Compiler Experiment Log

## Objective

Test whether a reusable token parser over frozen Qwen hidden states can recover
ordered program symbols and drive an executable latent modular program without
receiving token-span features at inference time.

## Primary Questions

1. Can shared token-level role and symbol heads recover initial values,
   operations, and arguments from the full hidden sequence?
2. Does a monotonic slot reader reuse the same parser across later operation
   steps instead of learning private per-step query slots?
3. Can trace bootstrap install an interface that generalizes to longer chains?
4. Does answer-only continuation preserve the installed parser interface?
5. Does answer-only training from scratch discover the interface?
6. How sensitive is the learned parser to wording shifts?

## Metrics

- `executor_accuracy`: accuracy after argmax compilation and exact execution.
- `executor_target_mass`: differentiable executor probability assigned to the
  target answer.
- `init_accuracy`: compiled initial value accuracy.
- `op_accuracy`: per-step operation accuracy.
- `arg_accuracy`: per-step argument accuracy.
- `program_exact`: fraction of examples with all compiled symbols correct.
- `direct_accuracy`: direct answer classifier accuracy from the frozen Qwen
  answer-marker feature.

## Artifact Layout

- Code and lightweight outputs:
  `experiments/qwen_shared_parser_compiler/`
- Checkpoints:
  `large_artifacts/qwen_shared_parser_compiler/checkpoints/`
- Run outputs:
  `experiments/qwen_shared_parser_compiler/runs/<run>/`
- Analysis outputs:
  `experiments/qwen_shared_parser_compiler/analysis/`

## 2026-06-21 Setup

Created the standalone experiment directory:

- `experiments/qwen_shared_parser_compiler/src/`
- `experiments/qwen_shared_parser_compiler/reports/`
- `experiments/qwen_shared_parser_compiler/runs/`
- `experiments/qwen_shared_parser_compiler/analysis/figures/`
- `large_artifacts/qwen_shared_parser_compiler/checkpoints/`

Implementation plan:

- Cache padded full-sequence hidden states and sequence masks.
- Replace private step queries with shared token role heads.
- Use a monotonic differentiable rank reader to convert role scores into
  ordered operation and argument slots.
- Keep direct, answer-only, trace, and trace-then-answer variants.
- Evaluate standard and paraphrase template splits across short and long
  program lengths.

## 2026-06-21 Harness Smoke

Implemented the shared-parser harness:

- Full hidden-sequence feature caching.
- Shared token role heads for init, operation, and argument tokens.
- Shared token symbol heads for residues and operation classes.
- Monotonic differentiable rank reader for ordered operation and argument
  slots.
- Trace-time selection loss over token roles and ordered slot positions.
- Staged direct, trace, answer-only, and trace-then-answer variants.
- External checkpoint writing.
- Analysis summary and figures.

Verification:

- Source compilation passed.
- Stale path and standalone wording scans passed.
- Tiny random Llama smoke completed.
- Analysis generation completed.
- Checkpoints were written under
  `large_artifacts/qwen_shared_parser_compiler/checkpoints/smoke_tiny/`.

Smoke iteration:

- The first smoke exposed unstable initial rank-reader losses because random
  role logits made cumulative role mass too large.
- Patched the parser to initialize role-head bias to `-2.5` and raised default
  rank temperature to `1.0`.
- `smoke_tiny_sparse_init` completed with sane loss scale.

Next action: run a Qwen trace pilot to test whether shared role parsing
generalizes from short trained chains to longer held-out chains.

## 2026-06-21 Qwen Pilot: Monotonic Shared Parser

Ran `pilot_qwen35_shared_parser_l4_trace` with frozen Qwen3.5-4B features,
standard-template length-4 trace training, and evaluation at lengths 4, 8, 12,
and 24 under standard and paraphrase templates.

Final standard-template metrics:

| Split | Exec | Init | Op | Arg | Program exact |
|---|---:|---:|---:|---:|---:|
| L=4 | 50.8% | 100.0% | 93.8% | 89.8% | 50.8% |
| L=8 | 3.1% | 100.0% | 77.1% | 58.1% | 0.0% |
| L=12 | 0.0% | 100.0% | 74.1% | 42.4% | 0.0% |
| L=24 | 0.0% | 100.0% | 69.8% | 21.5% | 0.0% |

Added explicit slot-position diagnostics and reran the matched
`pilot_qwen35_shared_parser_l4_trace_diag` condition for 400 steps. The
diagnostic result showed operation positions were essentially solved on
standard prompts, while argument positions drifted with sequence length:

| Split | Op pos | Arg pos | Op symbol | Arg symbol |
|---|---:|---:|---:|---:|
| L=4 | 100.0% | 90.8% | 97.9% | 89.3% |
| L=8 | 100.0% | 62.5% | 87.7% | 65.7% |
| L=12 | 100.0% | 45.2% | 88.2% | 48.6% |
| L=24 | 99.4% | 21.2% | 83.1% | 25.6% |

Interpretation:

- The init parser solved standard-template initial values.
- The operation slot reader generalized well on standard prompts.
- The independent monotonic argument reader was the length bottleneck.

## 2026-06-21 Iterations: Argument Reader

Added `role_count_loss_weight` to calibrate role counts. The matched
`pilot_qwen35_shared_parser_l4_count_trace` run did not improve argument
position drift; L24 argument-position accuracy remained about 21%.

Ran `pilot_qwen35_shared_parser_l12_trace` with trace training lengths 1-12.
This exposed later slots during training and improved argument localization, but
exact execution remained low because per-step errors compounded:

| Split | Exec | Op pos | Arg pos | Op symbol | Arg symbol |
|---|---:|---:|---:|---:|---:|
| L=4 | 50.0% | 100.0% | 93.6% | 98.2% | 85.9% |
| L=8 | 15.6% | 100.0% | 87.5% | 96.8% | 78.9% |
| L=12 | 3.1% | 100.0% | 78.4% | 96.0% | 70.8% |
| L=24 | 1.6% | 100.0% | 70.1% | 95.2% | 62.3% |

Increased argument trace and selection weights in
`pilot_qwen35_shared_parser_l12_arg4_trace`. This improved standard-template
execution to 75.8% at L=4, 39.1% at L=8, and 9.4% at L=12.

Added `arg_reader_mode=after_op`, which anchors each argument reader to the
learned operation slot and scores candidate argument tokens within a short
following window. The matched
`pilot_qwen35_shared_parser_l12_after_op_arg4_trace` run improved standard
L=12 exact execution to 18.0% and L=24 to 3.9%.

Ran a stronger trace pilot,
`pilot_qwen35_shared_parser_l12_after_op_strong_trace`, with 4096 trace
examples, width 768, and 2400 steps. The best intermediate checkpoint was at
step 1600:

| Step | L=4 exec | L=8 exec | L=12 exec | L=24 exec |
|---:|---:|---:|---:|---:|
| 800 | 77.7% | 59.4% | 31.6% | 1.2% |
| 1600 | 78.5% | 62.5% | 39.1% | 0.4% |
| 2400 | 80.1% | 50.4% | 28.5% | 1.2% |

Decision:

- Promote the after-op argument reader, argument-weighted trace loss, width
  768, 4096 trace examples, and 1600 trace steps to the main run.
- Keep L=24 and paraphrase splits as stress tests.
- Run direct and answer-only controls from scratch.
- Run answer-only retention variants to test whether final-answer continuation
  preserves the trace-installed parser.

## 2026-06-21 Main Qwen Trace/Control Run

Ran `main_qwen35_after_op_trace_controls` with frozen Qwen3.5-4B features,
training lengths 1-12, standard-template training, standard/paraphrase eval
splits, and 256 examples per eval split.

Final metrics:

| Variant | Split | L=4 | L=8 | L=12 | L=24 |
|---|---|---:|---:|---:|---:|
| `direct` | standard | 2.0% | 0.0% | 1.2% | 1.6% |
| `compiler_trace` | standard | 78.5% | 62.5% | 39.1% | 0.4% |
| `compiler_answer_only` | standard | 2.3% | 2.3% | 2.0% | 3.1% |
| `direct` | paraphrase | 1.2% | 0.8% | 2.0% | 1.2% |
| `compiler_trace` | paraphrase | 4.3% | 0.8% | 2.3% | 0.4% |
| `compiler_answer_only` | paraphrase | 1.2% | 1.6% | 1.2% | 1.6% |

Trace parser diagnostics on standard splits:

| Split | Init | Init pos | Op | Op pos | Arg | Arg pos | Program exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| L=4 | 100.0% | 100.0% | 99.9% | 100.0% | 94.0% | 95.1% | 78.1% |
| L=8 | 100.0% | 100.0% | 99.2% | 100.0% | 94.8% | 94.8% | 62.1% |
| L=12 | 100.0% | 100.0% | 99.0% | 100.0% | 93.2% | 93.7% | 38.7% |
| L=24 | 100.0% | 100.0% | 82.6% | 72.9% | 70.7% | 68.8% | 0.0% |

Interpretation:

- Trace supervision installs a useful standard-template parser up to the
  trained length range.
- Exact execution is limited by compounding per-step symbol errors.
- The parser does not extrapolate to length 24.
- The parser does not transfer to paraphrased prompt wording under this setup.
- Direct answer classification and answer-only parser training stay at chance.

## 2026-06-21 Main Qwen Retention Run

Ran `main_qwen35_after_op_retention` with the same parser recipe, then continued
training from the trace-installed state using final-answer loss only.

Final metrics:

| Variant | Split | L=4 | L=8 | L=12 | L=24 |
|---|---|---:|---:|---:|---:|
| `compiler_trace_then_answer` | standard | 60.2% | 0.4% | 1.2% | 0.8% |
| `compiler_trace_then_answer_low_lr` | standard | 78.1% | 6.6% | 0.8% | 0.0% |
| `compiler_trace_then_answer` | paraphrase | 3.1% | 2.0% | 2.0% | 1.2% |
| `compiler_trace_then_answer_low_lr` | paraphrase | 2.0% | 0.8% | 0.4% | 1.2% |

Retention interpretation:

- Normal-rate answer-only continuation rapidly destroys the length-general
  part of the installed parser.
- Low-rate continuation preserves the first post-switch step, but it also
  collapses longer standard lengths by the end of the answer stage.
- Final-answer continuation is not a reliable preservation or improvement
  method for this parser.

## 2026-06-21 Analysis Snapshot

Generated:

- `analysis/final_metrics.csv`
- `analysis/summary.md`
- `analysis/figures/direct_accuracy.png`
- `analysis/figures/executor_accuracy.png`
- `analysis/figures/program_exact.png`
- `checkpoint_manifest.csv`

Large checkpoint files are stored under
`large_artifacts/qwen_shared_parser_compiler/checkpoints/`.

## 2026-06-21 Final Audit

Final artifacts created:

- `reports/qwen_shared_parser_compiler_paper.md`
- `reports/qwen_shared_parser_compiler_paper.html`
- `checkpoint_manifest.csv`

Verification:

- Source compilation passed:
  `python -m py_compile src/qwen_shared_parser_compiler_experiment.py src/analyze_qwen_shared_parser_compiler.py`
- Analysis generation passed:
  `python src/analyze_qwen_shared_parser_compiler.py`
- Checkpoint manifest validation passed for 18 saved checkpoints.
- No `.pt`, `.pth`, or `.ckpt` files are stored inside the lightweight
  experiment directory.
- Standalone wording scan passed on the report files.
- Compile caches were removed after verification.

Artifact sizes:

- `experiments/qwen_shared_parser_compiler/`: 876K
- `large_artifacts/qwen_shared_parser_compiler/`: 99M

Conclusion:

The after-operation shared parser installs a useful standard-template latent
program interface under trace supervision, reaching 78.5% exact execution at
L=4, 62.5% at L=8, and 39.1% at L=12. It does not extrapolate to L=24, does
not transfer to paraphrase templates, and is not learned from answer-only
training. Answer-only continuation damages the installed parser on longer
standard lengths.
