# Qwen Context-Conditioned Trace Verifier

This standalone experiment tests whether a learned verifier can better select
hidden-VM candidate traces when it receives prompt-conditioned Qwen hidden-state
context in addition to candidate execution features.

Large checkpoints are stored outside this directory:

- `/workspace/large_artifacts/qwen_context_trace_verifier/checkpoints`

Local structure:

- `src/qwen_context_trace_verifier_experiment.py`: run generator, training, and evaluation
- `src/analyze_qwen_context_trace_verifier.py`: regenerate aggregate CSVs, figures, reports, and checkpoint manifest
- `runs/`: per-run logs and metrics
- `analysis/`: aggregate metrics and figures
- `reports/`: standalone Markdown and HTML report
- `checkpoint_manifest.csv`: checkpoint and artifact index

Regenerate analysis artifacts from completed runs:

```bash
python experiments/qwen_context_trace_verifier/src/analyze_qwen_context_trace_verifier.py
```
