# Qwen VM-ECHO Trace Distillation

This experiment tests whether a frozen-Qwen bytecode compiler improves when it
is trained not only to emit a target program, but also to predict the VM
observations produced by that program.

## Hypothesis

A compiler head attached to Qwen hidden states may learn more reusable program
semantics if the training signal includes the consequences of execution:
validity, final value, stack top after each slot, and stack depth after each
slot. The key comparison is a matched baseline versus a VM-ECHO arm with the
same Qwen features, same typed decoder, same answer-verified local search, and
same initialization.

## Layout

- `src/typed_bytecode_core.py`: standalone task generator, typed VM, decoder,
  candidate search, and utility functions.
- `src/qwen_vm_echo_trace_distillation_experiment.py`: frozen-Qwen feature
  extraction, compiler head, VM-ECHO losses, training, evaluation, and
  checkpoint manifest updates.
- `src/analyze_qwen_vm_echo_trace_distillation.py`: aggregation, charts, and
  Markdown/HTML report generation.
- `runs/`: per-run metrics, logs, and dataset manifests.
- `analysis/`: aggregated CSVs and generated figures.
- `reports/`: final Markdown and HTML writeups.
- `large_artifacts/qwen_vm_echo_trace_distillation/checkpoints/`: checkpoint
  files kept outside the experiment directory.

## Primary Metrics

- Direct executable accuracy from greedy decoded bytecode.
- Search/oracle accuracy from answer-verified candidate repair.
- Exact program match and validity.
- VM observation prediction accuracy: final value, validity, trace top, and
  trace depth.

