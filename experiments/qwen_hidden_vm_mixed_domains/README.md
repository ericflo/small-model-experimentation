# Qwen Hidden VM Mixed Domains

This experiment trains a Qwen-attached hidden virtual machine compiler on multiple deterministic task families. The model reads natural-language prompts, writes an invisible typed program into fixed slots, executes that program in a deterministic runtime, and returns the runtime answer.

Small files stay in this experiment directory. Large model adapters and head checkpoints are stored separately under:

```text
large_artifacts/qwen_hidden_vm_mixed_domains/checkpoints/
```

## Layout

- `src/qwen_hidden_vm_mixed_domains_experiment.py`: training and evaluation entrypoint.
- `analysis/analyze_qwen_hidden_vm_mixed_domains.py`: report, chart, and summary generator.
- `runs/`: run-local metrics, logs, and metadata.
- `reports/`: standalone markdown and HTML reports.
- `experiment_log.md`: running lab notebook.
- `checkpoint_manifest.csv`: generated manifest for large artifacts.

## Primary Question

Can a small QLoRA posttraining run teach a 4B Qwen model to compile several kinds of natural-language tasks into one hidden executable VM, rather than only learning a single modular-arithmetic grammar?
